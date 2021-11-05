from new_main import *

transform = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ])
}

path = path_new
select_fold = 1

data = ['original','balance_patch','non_balance_patch']

selected_data = data[0]

if selected_data == 'original':
    join = False
else:
    join = True

if path == 'new_data':
    path_mod = 'new/fold '+str(select_fold)+'/'
    path_save = naming_model('new',select_fold)
    kelas = getKelas(path)
    data_folder = get_folder(path,select_fold)
else:
    path_mod = 'old/fold '+str(select_fold)+'/'
    path_save = naming_model('old',select_fold)
    kelas = getKelas(path_old)
    data_folder = get_folder(path_old,select_fold)


print(kelas)
print(path_save)

data = {
    'train': data_folder[0][selected_data],
    'test': data_folder[1][selected_data],
}

batch_size = 100
pretrain = True
sampler = False
dropout = False
weight_decay = False
weight_entropy = False

data_loader = data_load(batch_size,
                        train=data['train'],
                        val=data['test'],
                        transform=transform,
                        sampler=sampler)

config = config_model(dropout=dropout, lr=0.001,
                      weight_decay=weight_decay,
                      kelas=kelas,
                      pretrained=pretrain,
                      data_loader=data_loader, freeze=False)

print('Validasi: ' + str(data_loader['sizes']['val']))
print('Training: ' + str(data_loader['sizes']['train']))

epoch = 60
to_train = False
check = True
num_mod = 16

#train model
if to_train:
    coba_train(config, transform,
               data_loader, epoch=epoch,
               pretrained=pretrain,
               sampler=sampler,
               dropout=dropout,
               batch_sizes=batch_size,
               sliced=data,
               weight_entropy=weight_entropy,
               weight_decay=weight_decay, path_save=path_save,
               join=join)
#check model
if check:
    check_model(path_mod,num_mod)
