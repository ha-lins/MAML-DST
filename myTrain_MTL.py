from tqdm import tqdm
import torch.nn as nn

from utils.config import *
from models.TRADE import *

'''
python myTrain.py -dec= -bsz= -hdd= -dr= -lr=
'''
ALL_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]
early_stop = args['earlyStop']

args["addName"] += "MTL"
if args['dataset'] == 'multiwoz':
    from utils.utils_multiWOZ_DST import *

    early_stop = None
else:
    print("You need to provide the --dataset information")
    exit(1)

# Configure models and load data
avg_best, cnt, acc = 0.0, 0, 0.0
train, dev, test, test_special, meta_lang, SLOTS_LIST, gating_dict, max_word = prepare_data_seq(True, args['task'], False,
                                                                                           batch_size=int(args['batch']))
source_domains = []

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
# print("[Info] Slots include ", SLOTS_LIST)
# print("[Info] Unpointable Slots include ", gating_dict)

avg_best, cnt, acc = 0.0, 0, 0.0

for epoch in range(200):
    print("Epoch:{}".format(epoch))
    source_train = []
    source_dev = []
    SOURCE_SLOTS_LIST = []
    init_state = copy.deepcopy(model.state_dict())

    for k in range(4):
    #for k-th task
    #sample tasks
        args['only_domain'] = source_domains[k]
        args['except_domain'] = ''

        train_single, dev_single, test_single, test_special, lang_single, SLOTS_LIST_single, gating_dict, max_word = prepare_data_seq(True, args['task'],False,
                                                                                                       batch_size=int(args['batch']))
        source_train.append(train_single)
        source_dev.append(dev_single)
        SOURCE_SLOTS_LIST.append(SLOTS_LIST_single)


    pbar = tqdm(enumerate(zip(source_train[0], source_train[1], source_train[2], source_train[3])), total=min(map(lambda single: len(single), source_train)))
    for i,data in pbar:
        loss_tasks = []

        for k in range(4):
        # for k-th task:
            model.load_state_dict(init_state)
            # Run the train function
            model.train_batch(data[k], int(args['clip']), SOURCE_SLOTS_LIST[k][1], reset=(i == 0))
            loss_tasks.append(model.loss_grad)

        model.load_state_dict(init_state)
        model.optimizer.zero_grad()
        model.loss_sum = torch.stack(loss_tasks).sum(0) / 4
        loss_tasks = []
        #model.optimize(args['clip'])
        model.loss_sum.backward()
        clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args['clip'])
        model.optimizer.step()
        pbar.set_description(model.print_loss())
        init_state = copy.deepcopy(model.state_dict())


    if ((epoch + 1) % int(args['evalp']) == 0):
        acc = model.evaluate(dev, avg_best, SLOTS_LIST[2], early_stop)
        model.scheduler.step(acc)
        if (acc >= avg_best):
            avg_best = acc
            cnt = 0
            best_model = model
        else:
            cnt += 1

        if (cnt == args["patience"] or (acc == 1.0 and early_stop == None)):
            print("Ran out of patient, early stop...")
            break

