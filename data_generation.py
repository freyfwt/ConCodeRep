import json
import os
import sys


def extract_fragment(file_path, method_name):
    if not os.path.exists(file_path):
        flag = False
        file_name = file_path.split('/')[-1]
        for root, dirs, files in os.walk('/'.join(file_path.split('/')[:3])):
            if file_name in files:
                file_path = root + '/' + file_name
                flag = True
                break
        if not flag:
            # print('路径不存在：', file_path)
            return
    with open(file_path, encoding='utf-8') as file:
        file_lines = file.readlines()
        num_lines = len(file_lines)
    ret = []
    for i, line in enumerate(file_lines):
        if line.find(method_name) != -1 \
                and ('static' in line
                     or 'final' in line
                     or 'public' in line
                     or 'protected' in line
                     or 'private' in line
                     or 'default' in line
                     or 'void' in line
                     or 'boolean' in line
                     or 'int' in line
                     or 'long' in line
                     or 'byte' in line
                     or line.strip().startswith('<T>')
                     or line.strip().startswith('/*non-public*/')
                     or line.strip()[0].isupper()) \
                and not (line.strip().startswith('/**')
                         or line.strip().startswith('*')
                         or line.strip().startswith('*/')
                         or line.strip().startswith('//')) \
                and ' new ' not in line \
                and not line.strip().endswith(';')\
                and line.find('(') > line.find(method_name):
            index = i - 1
            while index >= 0 and ('*' in file_lines[index] or '@' in file_lines[index]):
                index -= 1
            start_id = index + 1
            index = i
            stack = ['']
            while index < num_lines and len(stack) != 0:
                for char in file_lines[index]:
                    if char == '{':
                        if stack[0] == '':
                            stack[0] = '{'
                        else:
                            stack.append('{')
                    elif char == '}':
                        stack.pop(-1)
                        if len(stack) == 0:
                            break
                index += 1
            end_id = index
            ret_ins = ''.join(file_lines[start_id:end_id])
            ret.append(ret_ins)
    return '\n'.join(ret)


def get_context(callgraph_dir, project_name, class_name, method_name):
    with open(callgraph_dir + '/'+project_name+'.cg', 'r', encoding='utf-8') as cg_file:
        call_graph = cg_file.readlines()
    record = []
    for line in call_graph:
        if line.startswith('M') and class_name in line and method_name in line:
            line_list = line.split(' ')
            if class_name in line_list[0] and method_name in line_list[0]:
                if class_name not in line_list[1] or method_name not in line_list[1]:
                    record.append((1, line_list[1].strip()))
            elif class_name not in line_list[0] or method_name not in line_list[0]:
                record.append((0, line_list[0].strip()))
    record = list(set(record))
    return record


def parse_path(project: str, cg_node: str):
    root_dict = {'caffeine': 'caffeine/caffeine/src/main/java',
                 'checkstyle': 'checkstyle/src/main/java',
                 'commons-collections': 'commons-collections/src/main/java',
                 'commons-lang': 'commons-lang/src/main/java',
                 'commons-math': 'commons-math/src/main/java',
                 'deeplearning4j': 'deeplearning4j/nd4j/nd4j-common/src/main/java',
                 'eclipse.jdt.core': 'eclipse.jdt.core/org.eclipse.jdt.apt.core/src',
                 'freemind': 'freemind/freemind/freemind',
                 'guava': 'guava/guava/src',
                 'openjdk11': 'openjdk11/src',
                 'trove': 'trove/core/src/main/java'
                 }
    root = root_dict[project]
    file_raw_path = '/'.join(cg_node.strip('(M)').strip('M:').split(':')[0].split('.')) + '.java'
    file_path = root + '/' + file_raw_path
    method = cg_node.strip('(M)').strip('M:').split(':')[1].split('(')[0]
    return file_path, method


class Pipeline:
    def __init__(self, source_data_path, repos_dir, callgraph_dir, save_dir):
        self.source_data_path = source_data_path
        self.repos_dir = repos_dir
        self.callgraph_dir = callgraph_dir
        self.save_dir = save_dir
        self.dataset = None

    def extract_code(self):
        with open(self.source_data_path, 'r', encoding='utf-8') as source_data_file:
            self.dataset = json.load(source_data_file)

        length = len(self.dataset)
        idx = 0
        for data_ins in self.dataset:
            code1 = data_ins['first']
            code2 = data_ins['second']

            # code1
            file_path = '/'.join([self.repos_dir, code1['project'], code1['file']])
            method_name = code1['method'].split('(')[0].split('.')[-1]
            extracted_code1 = extract_fragment(file_path, method_name)
            code1['code'] = extracted_code1
            # print(code1['code'])

            # code2
            file_path = '/'.join([self.repos_dir, code2['project'], code2['file']])
            method_name = code2['method'].split('(')[0].split('.')[-1]
            extracted_code2 = extract_fragment(file_path, method_name)
            code2['code'] = extracted_code2
            # print(code2['code'])

            idx += 1
            print('\r', end='')
            print(f'{idx}/{length}', end='')
            sys.stdout.flush()
        return self.dataset

    def extract_context(self):
        length = len(self.dataset)
        for idx, sample in enumerate(self.dataset):
            sample_first = sample['first']
            sample_second = sample['second']

            project_name = sample_first['project']
            class_name = sample_first['method'].split('.')[0]
            method_name = sample_first['method'].split('.')[-1].split('(')[0]
            first_context_pointers = get_context(self.callgraph_dir, project_name, class_name, method_name)
            first_context_res = []
            for relation, node in first_context_pointers:
                context_path, context_name = parse_path(sample_first['project'], node)
                context_code = extract_fragment(self.repos_dir + '/' + context_path, context_name)
                if context_code:
                    first_context_res.append((relation, node, context_code))

            project_name = sample_second['project']
            class_name = sample_second['method'].split('.')[0]
            method_name = sample_second['method'].split('.')[-1].split('(')[0]
            second_context_pointers = get_context(self.callgraph_dir, project_name, class_name, method_name)
            second_context_res = []
            for relation, node in second_context_pointers:
                context_path, context_name = parse_path(sample_second['project'], node)
                context_code = extract_fragment(self.repos_dir + '/' + context_path, context_name)
                if context_code:
                    second_context_res.append((relation, node, context_code))

            if len(first_context_res) != 0 or len(second_context_res) != 0:
                sample_first['context'] = first_context_res
                sample_second['context'] = second_context_res

            print('\r', end='')
            print(f'{idx + 1}/{length}', end='')
            sys.stdout.flush()
        return self.dataset

    def save(self):
        with open(self.save_dir + '/context_dataset.json', 'w', encoding='utf-8') as save_file:
            json.dump(self.dataset, save_file, indent=4, ensure_ascii=False)
        return True

    def run(self):
        print('Start extracting the code of target method.')
        self.extract_code()
        print('Extraction ends.')
        print('Start extracting the code of context.')
        self.extract_context()
        print('Extraction ends.')
        self.save()
        print('The dataset with context has been saved.')


if __name__ == '__main__':
    SOURCE_DATA_PATH = './sesame/dataset.json'
    REPOS_DIR = './sesame/src/repos'
    CALLGRAPH_DIR = './callgraphs'
    SAVE_DIR = './data'

    ppl = Pipeline(SOURCE_DATA_PATH, REPOS_DIR, CALLGRAPH_DIR, SAVE_DIR)
    ppl.run()
