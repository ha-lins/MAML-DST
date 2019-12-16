from tqdm import tqdm
import torch.nn as nn

from utils.config import *
from models.TRADE import *

'''
python myTrain.py -dec= -bsz= -hdd= -dr= -lr=
'''
ALL_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]
early_stop = args['earlyStop']

args["addName"] += "MAML"
if args['dataset'] == 'multiwoz':
    from utils.utils_multiWOZ_DST import *

    early_stop = None
else:
    print("You need to provide the --dataset information")
    exit(1)

# Configure models and load data
avg_best, cnt, acc = 0.0, 0, 0.0
train, dev, test, test_special, meta_lang, SLOTS_LIST, gating_dict, max_word = prepare_data_seq(True, args['task'], False,
                                                                                           batch_size=int(
                                                                                               args['batch']))
source_domains = []
memory = {}
target_domain = args['except_domain']
ALL_DOMAINS.remove(target_domain)
source_domains = ALL_DOMAINS
model = globals()[args['decoder']](
    hidden_size=int(args['hidden']),
    lang=meta_lang,
    path=args['path'],
    task=args['task'],
    lr=float(args['learn']),
    lr_meta=float(args['learn_meta']),
    dropout=float(args['drop']),
    slots=SLOTS_LIST,
    gating_dict=gating_dict,
    nb_train_vocab=max_word)

dnd = DND(dict_len= 5)

avg_best, cnt, acc = 0.0, 0, 0.0

try:
    for epoch in range(200):
        print("Epoch:{}".format(epoch))
        source_train = []
        source_dev = []
        SOURCE_SLOTS_LIST = []
        meta_init_state = copy.deepcopy(model.state_dict())

        for k in range(4):
        #for k-th task
        #sample tasks
            args['only_domain'] = source_domains[k]
            args['except_domain'] = ''
            train_single, dev_single, test_single, test_special, lang_single, SLOTS_LIST_single, gating_dict, max_word = \
                prepare_data_seq(True, args['task'], False, batch_size=int(args['batch']))
            source_train.append(train_single)
            SOURCE_SLOTS_LIST.append(SLOTS_LIST_single)
            # train_single_resample, dev_single_resample, test_single_resample, test_special_resample, lang_single_resample, SLOTS_LIST_single_resample, gating_dict_resample, max_word_resample = \
            #     prepare_data_seq(True, args['task'], False, batch_size=int(args['batch']), seed=20)
            # source_train.append(train_single_resample)
            # SOURCE_SLOTS_LIST.append(SLOTS_LIST_single_resample)
            # print('[info] source_train is:{}  source_dev[k] is :{} zip(train, dev) is:{}'.format(source_train, source_dev, zip(source_train, source_dev)))

        pbar = tqdm(enumerate(zip(source_train[0], source_train[1], source_train[2], source_train[3])), total=min(map(lambda single: len(single), source_train)))
#                    source_train[4], source_train[5], source_train[6], source_train[7]
        for i, data in pbar:
            loss_tasks = []

            for k in range(4):
            # for k-th task:
                model.load_state_dict(meta_init_state)
                meta_init_state = copy.deepcopy(model.state_dict())

                # Run the train function
                # print('[info] len(data) is:{} '.format(len(data)))
                # j = 2*k
                model.train_batch(data[k], int(args['clip']), SOURCE_SLOTS_LIST[k][1], reset=(i == 0))
                model.optimize(args['clip'])
                pbar.set_description(model.print_loss())

                #resample the same data
                #loss for the meta-update
                # j = j+1
                model.train_batch(data[k], int(args['clip']), SOURCE_SLOTS_LIST[k][1], reset=(i == 0))

                loss_tasks.append(model.loss_grad)

            model.load_state_dict(meta_init_state)
            model.meta_optimizer.zero_grad()
            model.loss_meta = torch.stack(loss_tasks).sum(0) / 4
            model.loss_meta.backward()

            clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args['clip'])
            model.meta_optimizer.step()
            meta_init_state = copy.deepcopy(model.state_dict())

        if ((epoch + 1) % int(args['evalp']) == 0):
            acc = model.evaluate(dev, avg_best, SLOTS_LIST[2], early_stop)
            model.scheduler.step(acc)
            model.meta_scheduler.step(acc)

            if (acc >= avg_best):
                avg_best = acc
                cnt = 0
                weights_best = copy.deepcopy(model.state_dict())
            else:
                cnt += 1

            if (cnt == args["patience"] or (acc == 1.0 and early_stop == None)):
                print("Ran out of patient, early stop...")
                break
except KeyboardInterrupt:
    pass

key_emb = model.get_emb(SLOTS_LIST[0])
dnd.save_memory(key_emb, copy.deepcopy(model.state_dict()))

print('[info]fine-tuning on 4 domains')

BSZ = 8

args['only_domain'] = ''
args['except_domain'] = target_domain
args["data_ratio"] = 1
train_single, dev_single, test_single, _, _, SLOTS_LIST_single, _, _ = prepare_data_seq(True, args['task'], False, batch_size=BSZ)

avg_best, cnt, acc = 0.0, 0, 0.0

weights_best = copy.deepcopy(model.state_dict())

try:
    for epoch in range(100):
        print("Epoch:{}".format(epoch))
        # Run the train function
        pbar = tqdm(enumerate(train_single),total=len(train_single))
        for i, data in pbar:

            model.train_batch(data, int(args['clip']), SLOTS_LIST_single[1], reset=(i==0))
            model.optimize(args['clip'])
            pbar.set_description(model.print_loss())

        if((epoch+1) % int(args['evalp']) == 0):
            acc = model.evaluate(dev_single, avg_best, SLOTS_LIST_single[2], args["earlyStop"])
            model.scheduler.step(acc)
            if(acc > avg_best):
                avg_best = acc
                cnt=0
                weights_best = copy.deepcopy(model.state_dict())
            else:
                cnt+=1
            if(cnt == 6 or (acc==1.0 and args["earlyStop"]==None)):
                print("Ran out of patient, early stop...")
                break
except KeyboardInterrupt:
    pass

key_emb = model.get_emb(SLOTS_LIST_single[1])
dnd.save_memory(key_emb, copy.deepcopy(model.state_dict()))

directory = 'save/DND_dst-'+args["addName"]
if not os.path.exists(directory):
    os.makedirs(directory)
torch.save(dnd, directory + '/DND.pkl')

