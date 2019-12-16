from tqdm import tqdm
import torch.nn as nn

from utils.config import *
from utils.utils_multiWOZ_DST import collate_fn
# from models.NADST import *
from models.TRADE import *
from copy import deepcopy

'''
python myTrain.py -dec= -bsz= -hdd= -dr= -lr=
'''

#### LOAD MODEL path
except_domain = args['except_domain']
directory = args['path'].split("/")
HDD = directory[2].split('HDD')[1].split('BSZ')[0]
# decoder = directory[1].split('-')[0]
BSZ = int(args['batch']) if args['batch'] else int(directory[2].split('BSZ')[1].split('DR')[0])
args["decoder"] = "TRADE"
args["HDD"] = HDD
early_stop = args['earlyStop']

if args['dataset'] == 'multiwoz':
    from utils.utils_multiWOZ_DST import *

    early_stop = None
else:
    print("You need to provide the --dataset information")
    exit(1)

### LOAD DATA
train, dev, test, test_special, lang, SLOTS_LIST, gating_dict, max_word = prepare_data_seq(True, args['task'], False, batch_size=1,\
                                                                                           switch_exceptd=False)

args['only_domain'] = except_domain
args['except_domain'] = ''
args["fisher_sample"] = 0
args["data_ratio"] = 1
train_single, dev_single, test_single, _, _, SLOTS_LIST_single, _, _ = prepare_data_seq(True, args['task'], False, batch_size=1,\
                                                                                        switch_exceptd=False)
args['except_domain'] = except_domain


M = []
age = 0
# mem_len = args['memories']

model = globals()[args['decoder']](
    hidden_size=int(args['hidden']),
    lang=lang,
    path=args['path'],
    task=args['task'],
    lr=float(args['learn']),
    dropout=float(args['drop']),
    slots=SLOTS_LIST,
    gating_dict=gating_dict,
    nb_train_vocab=max_word)


def merge(sequences):
    '''
    merge from batch * sent_len to batch * max_len
    sequences: list of tensors
    '''
    lengths = [len(seq) for seq in sequences]
    max_len = 1 if max(lengths) == 0 else max(lengths)
    padded_seqs = torch.ones(len(sequences), max_len).long()
    for i, seq in enumerate(sequences):
        end = lengths[i]
        padded_seqs[i, :end] = seq[:end]
    padded_seqs = padded_seqs.detach()  # torch.tensor(padded_seqs)
    return padded_seqs, lengths


def merge_multi_response(sequences):
    '''
    merge from batch * nb_slot * slot_len to batch * nb_slot * max_slot_len
    sequences: list of lists [bsz, 30, slot_len]
    '''
    lengths = []
    seqs = []
    for bsz_seq in sequences:
        # lengths.append(list(bsz_seq.size())[1])
        seqs.append(bsz_seq)
        length = [len(v) for v in bsz_seq]
        lengths.append(length)
    max_len = max([max(l) for l in lengths])
    padded_seqs = []
    for bsz_seq in seqs:
        pad_seq = []
        for v in bsz_seq:
            v = v + [PAD_token] * (max_len - len(v))
            pad_seq.append(v)
        padded_seqs.append(pad_seq)
    padded_seqs = torch.tensor(padded_seqs)
    lengths = torch.tensor(lengths)
    return padded_seqs, lengths


def get_batch(point):
    data_batch = copy.deepcopy(point)
    if len(M) > 0:
        order = [i for i in range(0, len(M))]
        for j in range(1, args['replay_batch_size']):
            shuffle(order)
            k = order[j]
            sample = M[k]
            # print('=============M is {}'.format(M))
            # data_batch = merge_sample(sample)
            for k, v in data_batch.items():
                if torch.is_tensor(v):
                    if k == 'gating_label' or k == 'y_lengths' or k == 'turn_domain':
                        data_batch[k] = torch.cat((data_batch[k], sample[k]), 0)
                    else:
                        if j == 1:
                            data_batch[k] = [data_batch[k].squeeze(0)]
                            data_batch[k].append(sample[k].squeeze(0))
                        else:
                            data_batch[k].append(sample[k].squeeze(0))
                else:
                    data_batch[k].append(sample[k][0])
        # data_batch.sort(key=lambda x: len(x['context']), reverse=True)
        for k, v in data_batch.items():
            if k == 'context':
                data_batch[k], _ = merge(data_batch[k])
            if k == 'generate_y':
                data_batch[k], _ = merge_multi_response(map(lambda x: x.cpu().numpy().tolist(), v))

    return data_batch

avg_best, cnt, acc = 0.0, 0, 0.0
weights_best = deepcopy(model.state_dict())

try:
    for epoch in range(200):
        print("Epoch:{}".format(epoch))
        if epoch == 0:
            # Initialize buffer with old data
            print('[info] Initialize buffer with old data:')
            pbar = tqdm(enumerate(train), total=args['memories'])
            for i, old_data in pbar:
                if len(M) < int(args['init_old_ratio'] * args['memories']):
                    M.append(old_data)
                else:
                    break

        # Run the train function
        pbar = tqdm(enumerate(train_single), total=len(train_single))
        for i, data in pbar:  # steps through each data points
            age += 1
            before = copy.deepcopy(model.state_dict())
            for batch in range(0, args['batches_per_example']):
                weights_before = deepcopy(model.state_dict())
                # Draw batch from buffer:
                data_batch = get_batch(data)
                model.train_batch(data_batch, int(args['clip']), SLOTS_LIST[0], reset=(i == 0))
                model.optimize(args['clip'])
                pbar.set_description(model.print_loss())
                weights_after = model.state_dict()

                # Within batch Reptile meta-update:
                model.load_state_dict(
                    {name: weights_before[name] + ((weights_after[name] - weights_before[name]) * args['mer_beta']) for name in weights_before})

            after = model.state_dict()
            # Across batches Reptile meta-update:
            model.load_state_dict({name : before[name] + ((after[name] - before[name]) * args['mer_gamma']) for name in before})

            # Reservoir sampling memory update:
            if len(M) < args['memories']:
                M.append(data)

            else:
                p = random.randint(0, age)
                if p < args['memories']:
                    M[p] = data

        if ((epoch + 1) % int(args['evalp']) == 0):
            acc = model.evaluate(dev_single, avg_best, SLOTS_LIST_single[2], args["earlyStop"])
            model.scheduler.step(acc)
            if (acc >= avg_best):
                avg_best = acc
                cnt = 0
                weights_best = deepcopy(model.state_dict())
            else:
                cnt += 1
            if (cnt == 6 or (acc == 1.0 and args["earlyStop"] == None)):
                print("Ran out of patient, early stop...")
                break

except KeyboardInterrupt:
    pass

model.load_state_dict({name: weights_best[name] for name in weights_best})
model.eval()

# After Fine tuning...
print("[Info] After Fine Tune ...")
# print("[Info] Test Set on 4 domains...")
# acc_test_4d = model.evaluate(test_special, 1e7, SLOTS_LIST[2])
print("[Info] Test Set on 1 domain {} ...".format(except_domain))
acc_test = model.evaluate(test_single, 1e7, SLOTS_LIST[3])

