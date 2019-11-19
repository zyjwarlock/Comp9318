from data_process import Data_process, Vocab

data_process_ins = Data_process('src/class-0.txt', 'src/class-1.txt')
vocab = Vocab(data_process_ins.counter.vocabulary_, 10)
pass