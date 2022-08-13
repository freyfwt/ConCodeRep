import javalang
import json
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
        self.pair_ds = None
        self.train_pairs = None
        self.dev_pairs = None
        self.test_pairs = None

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

    def extract_pair(self):
        data_list = []
        confidence_map = {0: 0.6, 1: 0.8, 2: 1}
        tree_df = self.tree_ds
        id_list = list(tree_df['id'].values)
        for json_dict in self.dataset:
            accumulate = 0
            total_weight = 0
            for field in ['goals', 'operations', 'effects']:
                for rating_object in json_dict[field]:
                    if rating_object['rating'] != -1:
                        accumulate += rating_object['rating'] * confidence_map[rating_object['confidence']]
                        total_weight += confidence_map[rating_object['confidence']]
            score = round(accumulate / total_weight)
            data_ins = [
                int(json_dict['first']['dbid']),
                int(json_dict['second']['dbid']),
                score
            ]

            if data_ins[0] in id_list and data_ins[1] in id_list:
                data_list.append(data_ins)
        self.pair_ds = pd.DataFrame(data_list, columns=['id1', 'id2', 'label'])
        return self.pair_ds

    def split_data(self):
        data = self.pair_ds
        data_num = len(data)
        ratios = [int(r) for r in self.ratio.split(':')]
        train_split = int(ratios[0]/sum(ratios)*data_num)
        val_split = train_split + int(ratios[1]/sum(ratios)*data_num)

        data = data.sample(frac=1, random_state=self.seed)
        self.train_pairs = data.iloc[:train_split]
        self.dev_pairs = data.iloc[train_split:val_split]
        self.test_pairs = data.iloc[val_split:]

    def dictionary_and_embedding(self):
        pairs = self.train_pairs
        train_ids = pairs['id1'].append(pairs['id2']).unique()

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

    def merge(self, part):
        if part == 'train':
            pairs = self.train_pairs
        elif part == 'dev':
            pairs = self.dev_pairs
        else:
            pairs = self.test_pairs
        pairs['id1'] = pairs['id1'].astype(int)
        pairs['id2'] = pairs['id2'].astype(int)
        df = pd.merge(pairs, self.tree_ds, how='left', left_on='id1', right_on='id')
        df = pd.merge(df, self.tree_ds, how='left', left_on='id2', right_on='id')
        df.drop(['id_x', 'id_y'], axis=1, inplace=True)
        df.dropna(inplace=True)
        df.to_pickle(self.output_dir + f'/{part}_blocks.pkl')

    def run(self):
        self.extract_code_tree()
        self.extract_pair()
        self.split_data()
        self.dictionary_and_embedding()
        self.generate_block_seqs()
        self.merge('train')
        self.merge('dev')
        self.merge('test')


if __name__ == '__main__':
    DATA_PATH = './data/context_dataset.json'
    TREE_FILE_PATH = './data/trees.pkl'
    OUTPUT_DIR = './data/clone_detection'
    RATIO = '8:1:1'
    RANDOM_SEED = 2021
    EMBEDDING_SIZE = 128
    ppl = Pipeline(DATA_PATH, TREE_FILE_PATH, OUTPUT_DIR, RATIO, RANDOM_SEED, EMBEDDING_SIZE, tree_exists=True)
    ppl.run()
