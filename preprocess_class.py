import javalang
import json
import numpy as np
import os
import pandas as pd


class Pipeline:
    def __init__(self, data_path, tree_file_path, output_dir, ratio, random_seed, embedding_size, tree_exists=False):
        self.tree_exists = tree_exists
        self.tree_file_path = tree_file_path
        self.data_path = data_path
        self.output_dir = output_dir
        self.ratio = ratio
        self.seed = random_seed
        self.size = embedding_size
        self.dataset = None
        self.tree_ds = None
        self.labels = None
        self.train_trees = None
        self.dev_trees = None
        self.test_trees = None

    def extract_code_tree(self):
        with open(self.data_path, 'r', encoding='utf-8') as input_file:
            self.dataset = json.load(input_file)

        if self.tree_exists:
            if os.path.exists(self.tree_file_path):
                self.tree_ds = pd.read_pickle(self.tree_file_path)
                return self.tree_ds
            else:
                print('Warning: The path you specify to load tree dataset does not exist.')

        def process_context_code(code_object):
            def parse_program(func):
                tokens = javalang.tokenizer.tokenize(func)
                parser = javalang.parser.Parser(tokens)
                tree = parser.parse_member_declaration()
                return tree

            try:
                original_tree = parse_program(code_object['code'])
            except Exception:
                print(f"Warning: No. {code_object['dbid']} target cannot be parsed!")
                return code_object['dbid'], None, None, None
            calling_trees = []
            called_trees = []
            for tag, method, context_code in code_object['context']:
                try:
                    temp_tree = parse_program(context_code)
                    if tag == 0:
                        calling_trees.append(temp_tree)
                    elif tag == 1:
                        called_trees.append(temp_tree)
                except Exception:
                    print(f'Warning: The context {method} cannot be parsed!')
            return code_object['dbid'], original_tree, calling_trees, called_trees

        tree_array = []
        record = []
        for sample in self.dataset:
            for number in ['first', 'second']:
                code_ob = sample[number]
                if code_ob['dbid'] in record:
                    continue
                dbid, original_tree, calling_trees, called_trees = process_context_code(code_ob)
                tree_array.append([int(dbid), original_tree, calling_trees, called_trees])
                record.append(dbid)
        new_df = pd.DataFrame(tree_array, columns=['id', 'code', 'calling', 'called'])
        new_df = new_df.loc[pd.notnull(new_df['code']), ['id', 'code', 'calling', 'called']]
        self.tree_ds = new_df

        if not os.path.exists(os.path.dirname(self.tree_file_path)):
            os.mkdir(os.path.dirname(self.tree_file_path))
        self.tree_ds.to_pickle(self.tree_file_path)

        return self.tree_ds

    def split_data(self):
        data = self.tree_ds
        data_num = len(data)
        ratios = [int(r) for r in self.ratio.split(':')]
        train_split = int(ratios[0]/sum(ratios)*data_num)
        val_split = train_split + int(ratios[1]/sum(ratios)*data_num)

        data = data.sample(frac=1, random_state=self.seed)
        self.train_trees = data.iloc[:train_split]
        self.dev_trees = data.iloc[train_split:val_split]
        self.test_trees = data.iloc[val_split:]

    def dictionary_and_embedding(self):
        trees = self.train_trees
        train_ids = trees['id'].unique()

        trees = self.tree_ds.set_index('id', drop=False).loc[train_ids]
        from utils import get_sequence as func

        def trans_to_sequences(ast):
            sequence = []
            func(ast, sequence)
            return sequence

        trees_array = trees.values
        corpus = []
        for i, tree_sample in enumerate(trees_array):
            for calling_tree in tree_sample[2]:
                ins_seq = trans_to_sequences(calling_tree)
                corpus.append(ins_seq)

            ins_seq = trans_to_sequences(tree_sample[1])
            corpus.append(ins_seq)

            for called_tree in tree_sample[3]:
                ins_seq = trans_to_sequences(called_tree)
                corpus.append(ins_seq)

        from gensim.models.word2vec import Word2Vec
        w2v = Word2Vec(corpus, size=self.size, workers=16, sg=1, max_final_vocab=3000)
        w2v.save(self.output_dir+'/node_w2v_' + str(self.size))

    def generate_block_seqs(self):
        from utils import get_blocks_v1 as func
        from gensim.models.word2vec import Word2Vec

        word2vec = Word2Vec.load(self.output_dir+'/node_w2v_' + str(self.size)).wv
        vocab = word2vec.vocab
        max_token = word2vec.syn0.shape[0]

        def tree_to_index(node):
            token = node.token
            result = [vocab[token].index if token in vocab else max_token]
            children = node.children
            for child in children:
                result.append(tree_to_index(child))
            return result

        def trans2seq(r):
            blocks = []
            func(r, blocks)
            tree = []
            for b in blocks:
                btree = tree_to_index(b)
                tree.append(btree)
            return tree

        def trans2seqs(r):
            ret = []
            for ins_r in r:
                tree = trans2seq(ins_r)
                ret.append(tree)
            return ret

        trees = pd.DataFrame(self.tree_ds, copy=True)
        trees['code'] = trees['code'].apply(trans2seq)
        trees['calling'] = trees['calling'].apply(trans2seqs)
        trees['called'] = trees['called'].apply(trans2seqs)

        # Save only the longest context
        trees_array = trees.values
        # trees_list = []
        for block_sample in trees_array:
            max_tree_length = 0
            max_tree = []
            for calling_tree in block_sample[2]:
                tree_length = sum([len(statement) for statement in calling_tree])
                if tree_length > max_tree_length:
                    max_tree = calling_tree
                    max_tree_length = tree_length
            block_sample[2] = max_tree
            max_tree_length = 0
            max_tree = []
            for called_tree in block_sample[3]:
                tree_length = sum([len(statement) for statement in called_tree])
                if tree_length > max_tree_length:
                    max_tree = called_tree
                    max_tree_length = tree_length
            block_sample[3] = max_tree
            # trees_list.append(list(block_sample))

        # with open('./data/sesame_tokens_with_context.json', 'w', encoding='utf-8') as out_file:
        #     json.dump(trees_list, out_file, indent=4)
        trees = pd.DataFrame(trees_array, columns=['id', 'code', 'calling', 'called'])
        self.tree_ds = trees

    def generate_class_ds(self):
        class_data = []
        for pair in self.dataset:
            sample = [int(pair['first']['dbid']), pair['first']['project']]
            class_data.append(sample)
            sample = [int(pair['second']['dbid']), pair['second']['project']]
            class_data.append(sample)
        class_data = sorted(class_data, key=lambda x: x[0])
        class_data_df = pd.DataFrame(class_data, columns=['id', 'label'])
        class_data_df.drop_duplicates(subset=['id'], keep='first', inplace=True)
        # You can uncomment the following statement to save the class label file.
        # class_data_df.to_csv('./data/classification/class_label.csv', index=False)

        classes = np.unique(class_data_df['label'].values)
        classes = sorted(classes)
        classes_map = {name: i for i, name in enumerate(classes)}
        class_data_df['label'] = class_data_df['label'].apply(lambda x: classes_map[x])
        self.labels = class_data_df

        train_df = pd.merge(self.train_trees[['id']], self.tree_ds, how='left', left_on='id', right_on='id')
        dev_df = pd.merge(self.dev_trees[['id']], self.tree_ds, how='left', left_on='id', right_on='id')
        test_df = pd.merge(self.test_trees[['id']], self.tree_ds, how='left', left_on='id', right_on='id')
        train_df = pd.merge(train_df, class_data_df, how='left', left_on='id', right_on='id')
        dev_df = pd.merge(dev_df, class_data_df, how='left', left_on='id', right_on='id')
        test_df = pd.merge(test_df, class_data_df, how='left', left_on='id', right_on='id')

        train_df.to_pickle(self.output_dir + '/train_df.pkl')
        dev_df.to_pickle(self.output_dir + '/dev_df.pkl')
        test_df.to_pickle(self.output_dir + '/test_df.pkl')
        return True

    def generate_random_class_ds(self):
        final_df = pd.merge(self.tree_ds, self.labels, how='left', left_on='id', right_on='id')

        for i in range(11):
            calling_series = final_df.loc[final_df['label'] == i, 'calling']
            called_series = final_df.loc[final_df['label'] == i, 'called']
            calling_random_series = calling_series.sample(frac=1, random_state=self.seed)
            called_random_series = called_series.sample(frac=1, random_state=self.seed)
            final_df.loc[final_df['label'] == i, 'calling'] = calling_random_series.values
            final_df.loc[final_df['label'] == i, 'called'] = called_random_series.values

        train_df = pd.merge(self.train_trees[['id']], final_df, how='left', left_on='id', right_on='id')
        dev_df = pd.merge(self.dev_trees[['id']], final_df, how='left', left_on='id', right_on='id')
        test_df = pd.merge(self.test_trees[['id']], final_df, how='left', left_on='id', right_on='id')

        train_df.to_pickle(self.output_dir + '/train_random_df.pkl')
        dev_df.to_pickle(self.output_dir + '/dev_random_df.pkl')
        test_df.to_pickle(self.output_dir + '/test_random_df.pkl')
        return True

    def run(self):
        self.extract_code_tree()
        self.split_data()
        self.dictionary_and_embedding()
        self.generate_block_seqs()
        self.generate_class_ds()
        self.generate_random_class_ds()


if __name__ == '__main__':
    DATA_PATH = './data/context_dataset.json'
    TREE_FILE_PATH = './data/trees.pkl'
    OUTPUT_DIR = './data/classification'
    RATIO = '8:1:1'
    RANDOM_SEED = 2021
    EMBEDDING_SIZE = 128
    ppl = Pipeline(DATA_PATH, TREE_FILE_PATH, OUTPUT_DIR, RATIO, RANDOM_SEED, EMBEDDING_SIZE, tree_exists=True)
    ppl.run()
