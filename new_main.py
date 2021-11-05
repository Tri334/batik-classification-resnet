import copy

import cv2
import math
import os
import random
import shutil as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import itertools

from torch.utils.data import WeightedRandomSampler, DataLoader
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

path_old = 'old_data'
path_new = 'new_data'

folder = ['original', 'balance_patch', 'non_balance_patch']
sub_fold = ['train', 'test']

# Mendapatkan folder untuk training
def get_folder(used_data,fold,folder=folder):
    if used_data == 'new_data':
        used_data='new'
    else: used_data = 'old'
    fold = str(fold)
    train = {
        'original': used_data +'/fold '+ fold + '/'+ folder[0] +'/train',
        'balance_patch': used_data +'/fold '+ fold+ '/'+ folder[1] +'/train',
        'non_balance_patch': used_data +'/fold '+ fold+ '/'+ folder[2] +'/train'
    }
    test = {
        'original': used_data +'/fold '+ fold+ '/'+ folder[0] +'/test',
        'balance_patch': used_data +'/fold '+ fold+ '/'+ folder[1] +'/test',
        'non_balance_patch': used_data +'/fold '+ fold+ '/'+ folder[2] +'/test'
    }
    return train,test

# Menamai model
def naming_model(used_data,fold):
    path = used_data +'/'+'fold '+str(fold)
    if not os.path.isdir(path+'/model'):
        os.makedirs(path+'/model')
    if len(os.listdir(path+'/model')) == 0:
        counter = 1
    else:
        counter = len(os.listdir(path+'/model')) + 1

    path_save = path+'/model/'+str(counter) + '.pth'
    return path_save

# Mengambil kelas
def getKelas(path):
    if path == 'new_data':
        kelas = os.listdir('new_data/')
    else:
        kelas = os.listdir('old_data/')
    return kelas

# Membuat folder
def make_folder(path, folder, kelas, fold, sub_fold=sub_fold):
    fold = str(fold+1)
    for sub in sub_fold:
        for item in kelas:
            if not os.path.isdir(path + '/' + 'fold '+ fold + '/' + folder + '/' + sub + '/' + item):
                os.makedirs(path + '/' + 'fold '+ fold + '/' + folder + '/' + sub + '/' + item)
            else:
                st.rmtree(path + '/' + 'fold '+ fold + '/' + folder + '/' + sub + '/' + item)
                os.makedirs(path + '/' + 'fold '+ fold + '/' + folder + '/' + sub + '/' + item)

# Membuat 5 fold ambil rate 20% val
def make_fold(used_data,kelas, folder=folder, fold=5):
    # Cek ini
    if used_data == 'old_data':
        dst_n = 'old'
        src_n = 'old_data'
    else:
        dst_n = 'new'
        src_n = 'new_data'

    folder=folder[0]

    dataset = datasets.ImageFolder(src_n)
    #Jumlah data tiap kelas
    sum_label = [0, 0, 0, 0, 0]
    for gambar, label in dataset.imgs:
        sum_label[label] += 1

    #Jumlah item yang dibutuhkan tiap kelas
    needed = [math.floor(i * 0.2) for i in sum_label]
    needed = {i: needed[i] for i in range(len(needed))}

    #Mendapatkan idx dari tiap kelas secara acak sesuai jumlah item yang dibutuhkan
    taken = [[], [], [], [], []]
    for fld in range(fold):
        # Membuat folder dan menghapus folder lama
        make_folder(dst_n, folder, kelas,fld)
        for kls in needed:
            cek = os.listdir(src_n + '/' + dataset.classes[kls])
            must_taken = [i for i in range(len(cek))]
            random.shuffle(must_taken)
            temp = []
            counter = 0
            #Perulangan sebanyak item yang dibutuhkan per kelas
            while len(temp) < needed[kls]:
                if must_taken[counter] not in taken[kls]:
                    # print('ok')
                    temp.append(must_taken[counter])
                    taken[kls].append(must_taken[counter])
                counter += 1

            # Memindahkan file
            for i in range(len(cek)):
                if i in temp:
                    test = 'test'
                else:
                    test = 'train'
                src = src_n + '/' + dataset.classes[kls] + '/' + cek[i]
                desti = dst_n + '/' + 'fold ' \
                        + str(fld + 1) + '/' + folder + '/' \
                        + test + '/' + dataset.classes[kls] \
                        + '/' + cek[i]
                st.copyfile(src, desti)

# Membuat balance patch
def patch_balance(used_data,kelas,folder=folder,expected_balance=1500,fold=5):
    folder_dst = folder[1]
    folder_src = folder[0]
    if used_data == 'new_data':
        used_data = 'new'
    else:
        used_data = 'old'

    for jumlah_fold in range(fold):
        path = used_data + '/' + 'fold ' + str(jumlah_fold + 1) + '/' + folder_src + '/' + 'train'
        print(path)
        sum_label = [0, 0, 0, 0, 0]

        for gambar, label in datasets.ImageFolder(path).imgs:
            sum_label[label] += 1

        expected_balanced = expected_balance / len(kelas)
        # print(expected_balance)
        # print(needed)
        needed = []
        for item in sum_label:
            temp = 0
            temp2 = 0
            while temp2 < expected_balanced:
                item -= 1
                temp += 1
                temp2 = temp * 4 + item
            needed.append(temp)
        slice2 = {kelas[i]: needed[i] for i in range(len(kelas))}
        # Check if possible
        # print(slice2)
        # print(needed)

        check = [sum_label[i] - needed[i] for i in range(len(kelas))]
        # checking
        sub_fold = ['train', 'test']

        possible = True
        for item in check:
            if item < 0:
                possible = False
                print(check)

        if possible:
            print('Requirement is met ✔')
            print('Getting Ready...\n')
            make_folder(used_data, folder_dst, kelas, jumlah_fold)

            num_slices_per_axis = 2
            # print(path)

            source_slice = {}
            source_original = {}
            for train in sub_fold:
                # print(train)
                iter = 0
                if train == 'train':
                    pathx = used_data + '/' + 'fold '+str(jumlah_fold+1) + '/' + 'original' + '/' + train + '/'
                    # print(pathx)
                    for kls in kelas:
                        # print(kls)
                        img = os.listdir(pathx + kls)
                        original_temp = [x for x in img]
                        slice_temp = []
                        for i in range(slice2[kls]):
                            try:
                                ran = random.randint(0, len(original_temp) - 1)
                                test2 = original_temp[ran]
                                if test2 not in slice_temp:
                                    # print(test2)
                                    slice_temp.append(original_temp.pop(ran))
                                    # counter += 1
                            except:
                                print('')
                                # print(len(original_temp))
                        source_slice[kls] = slice_temp
                        source_original[kls] = original_temp
                    tesx = {'slice': source_slice,'original': source_original}

                    #Memindahkan data
                    for orislice in tesx:
                        for kls in kelas:
                            if orislice == 'slice':
                                for item in source_slice[kls]:
                                    img = cv2.imread(pathx + kls + '/' + item)
                                    try:
                                        slice_shape = (int(img.shape[0] / num_slices_per_axis), int(img.shape[1] / num_slices_per_axis))
                                        for i in range(num_slices_per_axis):
                                            for j in range(num_slices_per_axis):
                                                top_left = (int(i * slice_shape[0]), int(j * slice_shape[1]))
                                                crop = img[top_left[0]:(top_left[0] + slice_shape[0]),
                                                       top_left[1]:(top_left[1] + slice_shape[1])]
                                                # Menaruh hasil pembelahan image
                                                dst = used_data + '/' + 'fold '+str(jumlah_fold+1) + '/' + folder_dst + '/' + train + '/' \
                                                      + kls + '/' + (str(iter)) + '_' + kls + '.jpg'
                                                # print(dst)
                                                # print(dst)
                                                cv2.imwrite(dst, crop)
                                                iter+=1
                                    except: print(img)
                            elif orislice == 'original':
                                for item in source_original[kls]:
                                    src = used_data + '/' + 'fold '+str(jumlah_fold+1) + '/' + folder_src + '/' + train + '/' + kls + '/' + item
                                    desti = used_data + '/' + 'fold '+str(jumlah_fold+1) + '/' + folder_dst + '/' + train + '/' + kls + '/' + item
                                    try:
                                        # print(src)
                                        # print('')
                                        st.copyfile(src, desti)
                                    except:
                                        # print(src)
                                        print('')
                        # print('Done ')
                else:
                    # print(fold)
                    dat = datasets.ImageFolder(used_data + '/' + 'fold '+str(jumlah_fold+1) + '/' + folder_src+'/'+train)
                    for image in dat.imgs:
                        # print(dat.classes[image[1]])
                        img = cv2.imread(image[0])
                        try:
                            slice_shape = (int(img.shape[0] / num_slices_per_axis), int(img.shape[1] / num_slices_per_axis))
                            for i in range(num_slices_per_axis):
                                for j in range(num_slices_per_axis):
                                    top_left = (int(i * slice_shape[0]), int(j * slice_shape[1]))
                                    crop = img[top_left[0]:(top_left[0] + slice_shape[0]),
                                           top_left[1]:(top_left[1] + slice_shape[1])]
                                    # Menaruh hasil pembelahan image
                                    cv2.imwrite(
                                        used_data +'/'+ 'fold '+str(jumlah_fold+1) + '/' + folder_dst + '/'
                                        + train + '/' + str(dat.classes[image[1]]) + '/' + str(iter) + '_' +
                                        str((dat.classes[image[1]]) + '.jpg'), crop)
                                    iter += 1
                                    # print(iter)
                        except:
                            # print(img)
                            print('')
        else: print('## Try lowering expected total sliced data ##\n')

# Membuat non balance patch
def patch_non_balance(used_data,kelas,folder=folder, sub_fold=sub_fold,fold=5):
    # Penyalinan image sliced ke folder baru
    folder_dst = folder[2]
    folder_src = folder[0]

    if used_data == 'new_data':
        used_data = 'new'
    else:
        used_data = 'old'

    for jumlah_fold in range(fold):

        path = used_data + '/' + 'fold ' + str(jumlah_fold + 1) + '/' + folder_src + '/'

        make_folder(used_data, folder_dst, kelas, jumlah_fold)
        num_slices_per_axis = 2
        iter = 0

        for train in sub_fold:
            dat = datasets.ImageFolder(path + '/' + train)
            print(dat)
            for image in dat.imgs:
                # print(dat.classes[image[1]])
                img = cv2.imread(image[0])
                try:
                    slice_shape = (int(img.shape[0] / num_slices_per_axis), int(img.shape[1] / num_slices_per_axis))
                    for i in range(num_slices_per_axis):
                        for j in range(num_slices_per_axis):
                            top_left = (int(i * slice_shape[0]), int(j * slice_shape[1]))
                            crop = img[top_left[0]:(top_left[0] + slice_shape[0]),top_left[1]:(top_left[1] + slice_shape[1])]
                            # Menaruh hasil pembelahan image
                            cv2.imwrite(
                                used_data + '/' + 'fold ' + str(jumlah_fold + 1) + '/' + folder_dst + '/'
                                + train + '/' + str(dat.classes[image[1]]) + '/' + str(iter) + '_' +
                                str((dat.classes[image[1]]) + '.jpg'), crop)
                            iter += 1
                            # print(iter)
                except:
                    print(img)




#Loader data
def data_load(batch_size, transform, train, val, sampler=False):
    # About batch size
    # https://discuss.pytorch.org/t/i-get-a-much-better-result-with-batch-size-1-than-when-i-use-a-higher-batch-size/20477/4
    # print(train,val,test)

    train_s = datasets.ImageFolder(train, transform=transform['train'])
    val_s = datasets.ImageFolder(val, transform=transform['val'])
    # test_s = datasets.ImageFolder(test, transform=transform['val'])

    train_load_s = DataLoader(train_s, shuffle=True, batch_size=batch_size,pin_memory=True)
    if sampler:
        samplerx = Wightedrandomsampler(train_s)
        train_load_s = DataLoader(train_s, sampler=samplerx['sampler'], batch_size=batch_size,pin_memory=True)
    val_load_s = DataLoader(val_s, shuffle=False, batch_size=batch_size,pin_memory=True)
    # test_load_s = DataLoader(test_s,shuffle=False, batch_size=batch_size,pin_memory=True)
    image_folder = {
        'train': train_s,
        'val': val_s,
        # 'test': test_s
    }
    loader_data = {
        'train': train_load_s,
        'val': val_load_s,
        # 'test': test_load_s
    }
    dataset_sizes = {
        'train': len(train_s),
        'val': len(val_s),
        # 'test': len(test_s)
    }
    load_data = {
        'loader': loader_data,
        'sizes': dataset_sizes,
        'image': image_folder
    }
    return load_data

# Membuat image agar bisa di petakan
def display(image):
    img = image
    img = img.numpy()
    img = np.transpose(img, (1, 2, 0))
    return img

# Membuat gambar hasil transfor bisa dilihat mata
def display_image(dataset):
    fig, axis = plt.subplots(3, 3, figsize=(15, 10))
    for i, ax in enumerate(axis.flat):
        with torch.no_grad():
            ran = random.randint(1, len(dataset) - 1)
            image, label = dataset[ran][0], dataset.classes[dataset[ran][1]]
            title = label
            ax.imshow(display(image))
            ax.set(title=title)
    plt.show()

# Membuat sampler untuk membalancing data
def Wightedrandomsampler(Dataset):
    root = Dataset
    sum_label = [0, 0, 0, 0, 0]
    for gambar, label in root.imgs:
        sum_label[label] += 1
    weight = [1. / x for x in sum_label]
    weight = torch.FloatTensor(weight)
    samples_weight = [weight[t].to(device) for t in root.targets]
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    sampler = {
        'weight': samples_weight,
        'sampler' : sampler
    }
    return sampler




def config_model(dropout,lr,weight_decay,data_loader,kelas,pretrained=False,weight_entropy = False,freeze=False):
    model = models.resnet18(pretrained=pretrained)
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
    feature = model.fc.in_features
    model.fc = torch.nn.Linear(feature, len(kelas))
    if dropout:
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.ReLU(),
            torch.nn.Linear(feature,len(kelas))
        )
    model = model.to(device)

    # config criterion, optim, lr
    criterion = torch.nn.CrossEntropyLoss()
    if weight_entropy:
        samplerx = Wightedrandomsampler(data_loader['image']['train'])
        criterion = torch.nn.CrossEntropyLoss(weight=samplerx['weight'])
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    if weight_decay:
        optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    expLr_scheduler = lr_scheduler.StepLR(optim, step_size=30, gamma=0.1)

    config={'model': model,
            'criterion': criterion,
            'optim': optim,
            'lr_scheduler': expLr_scheduler}

    return config




def eval(data_loader,config,join=False):
    counter = 0
    counter2 = 0
    add = [0]
    all_classx = torch.tensor([]).to(device)
    new_masukan = torch.tensor([]).to(device)
    with torch.no_grad():
        config['model'].eval()
        for i, (masukan, all_class) in enumerate(data_loader['loader']['val']):
            masukan = masukan.to(device)
            keluaran = config['model'](masukan)
            # config['optim'].zero_grad()
            if join:
                for item in all_class:
                    counter += 1
                    if counter == 4:
                        x = item.view(1)
                        all_classx= torch.cat((all_classx, x.to(device)),0)
                        counter=0

                for item in keluaran:
                    if counter2 != 4:
                        counter2 += 1
                        add[0] += item
                        add[0] = add[0].view(1,5)
                        # print(item)
                        if counter2 == 4:
                            new_masukan = torch.cat((new_masukan,add[0]),0)
                            # print('='*5)
                            # print(add[0])
                            # print('='*5)
                            add[0]=0
                            counter2=0
            else:
                new_masukan = torch.cat((new_masukan,keluaran),0)
                all_classx = torch.cat((all_classx,all_class.to(device)),0)

    eval ={
        'prediksi':new_masukan,
        'kelas' : all_classx
    }
    return eval

def conf_matrix(all_class,keluaran,kelas):
    confusion_matrix = torch.zeros(len(kelas), len(kelas))
    correct_pred = {classname: 0 for classname in kelas}
    total_pred = {classname: 0 for classname in kelas}
    _, prediksi = torch.max(keluaran, 1)

    for asli, pred in zip(all_class.view(-1), prediksi.view(-1)):
        confusion_matrix[asli.int(), pred.int()] += 1

    for label, predik in zip(all_class.int(), prediksi):
        if label == predik:
            correct_pred[kelas[label]] += 1
        total_pred[kelas[label]] += 1

    all_acc = []
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        all_acc.append(accuracy)
    tot_acc = sum(all_acc) / len(kelas)
    eval = {
        'matrix': confusion_matrix,
        'acc': tot_acc
    }
    return eval




def coba_train(config, transform, data_loader, epoch,pretrained,sampler, dropout, batch_sizes,sliced,weight_entropy,weight_decay,path_save, device=device ,join=False):

    waktu_mulai = time.time()
    val_acc = []
    loss_var = []
    test_acc = []
    model_terbaik = copy.deepcopy(config['model'].state_dict())
    akurasi_terbaik = 0.0
    loss_terbaik = 0.0

    for poch in range(epoch):
        print('Epoch {}/{}'.format(poch, epoch - 1))
        print('Lr:', config['optim'].param_groups[0]['lr'])
        print(10 * '=')

        # fase ada dua yaitu train dan validasi
        for fase in ['train', 'val']:
            if fase == 'train':
                # Fase training
                config['model'].train()
            else:
                # Fase evaluasi
                config['model'].eval()

            loss_saatIni = 0.0
            akurasi_saatIni = 0.0

            # Iterasi data
            for masukan, label in data_loader['loader'][fase]:
                masukan = masukan.to(device)
                label = label.to(device)

                # optime zero params gradient untuk backpropagasi
                config['optim'].zero_grad()

                # Propagasi maju
                with torch.set_grad_enabled(fase == 'train'):
                    keluaran = config['model'](masukan)
                    _, prediksi = torch.max(keluaran, 1)
                    loss = config['criterion'](keluaran, label)

                    # Propagasi balik + lr step
                    if fase == 'train':
                        loss.backward()
                        config['optim'].step()

                # Untuk statistik
                loss_saatIni += loss.item() * masukan.size(0)
                akurasi_saatIni += torch.sum(prediksi == label.data)

            if fase == 'train':
                config['lr_scheduler'].step()

            epoch_loss = loss_saatIni / data_loader['sizes'][fase]
            loss_var.append(epoch_loss)
            epoch_acc = akurasi_saatIni / data_loader['sizes'][fase]
            val_acc.append(epoch_acc)


            print('{} Loss: {:.5f} Akurasi: {:.5f}'.format(
                fase, epoch_loss, epoch_acc
            ))

            # Menyalin model
            if fase == 'val' and epoch_acc > akurasi_terbaik:
                akurasi_terbaik = epoch_acc
                loss_terbaik = epoch_loss
                model_terbaik = config['model'].state_dict()
                optimx = config['optim'].state_dict()
                print('Best Val Test Acc ✔')

        print()

    waktu_selesai = time.time() - waktu_mulai
    print('Training selesai pada {:.0f} menit {:.0f} detik'.format(
        waktu_selesai // 60, waktu_selesai % 60
    ))
    print('Validasi Terbaik: {:3f}'.format(akurasi_terbaik))
    print('Dengan Loss: {:3f}'.format(loss_terbaik))

    # Menyimpan model, epoch, loss, dll

    torch.save({
        'epoch': epoch,
        'optim': optimx,
        'model': model_terbaik,
        'loss': loss_terbaik,
        'acc': akurasi_terbaik,
        'transform': transform,
        'loss_plot': loss_var,
        'acc_plot': val_acc,
        'test_plot': test_acc,
        'batch_sizes': batch_sizes,
        'pretrain': pretrained,
        'using_sampler': sampler,
        'dropout': dropout,
        'slice': sliced,
        'weight_entropy': weight_entropy,
        'decay': weight_decay,
        'data_loader':data_loader,
        'join': join,
        'config': config}, path_save)


    # model.load_state_dict(model_terbaik)
    # return model,loss_var,val_acc




def plot_matrix(confusion_matrix,tot_acc,path,nama_plot,kelas):
    # Membuat confusion matrix
    if not os.path.isdir(path+'/plot_matrix'):
        os.makedirs(path+'/plot_matrix')
    pathx = path+'/plot_matrix/'
    title = 'Confusion Matrix | rata-rata: {}% '.format(tot_acc)
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(kelas))
    plt.xticks(tick_marks, kelas, rotation=0)
    plt.yticks(tick_marks, kelas, rotation=0)

    fmt = '.2f'
    threshold = confusion_matrix.max() / 2.
    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(j, i, format(confusion_matrix[i, j], fmt), horizontalalignment="center",
                 color='white' if confusion_matrix[i, j] > threshold else "black")

    plt.savefig(pathx + str(nama_plot) + '.png')
    plt.clf()

def plot_model(model_loss, model_acc,nama_plot,path):
    if not os.path.isdir(path+'/plot_acc_loss'):
        os.makedirs(path+'/plot_acc_loss')

    path_plot = path+'/plot_acc_loss/'
    train_loss = model_loss[::2]
    val_loss = model_loss[1::2]

    temp = []
    for item in model_acc:
        temp.append(item.item())

    train_acc = temp[::2]
    vals_acc = temp[1::2]

    # Saatnya plot perbandingan

    # plot perbandingan Loss
    plt.title('Perbandingan Akurasi')
    plt.plot(train_acc, label='Train Acc')
    plt.plot(vals_acc, label='Val Acc')
    plt.xlabel("Epoch")
    plt.ylabel("Akurasi")
    plt.legend()
    # plt.show()
    plt.savefig(path_plot +'Akurasi_' +str(nama_plot) + '.png')
    plt.clf()

    plt.title('Perbandingan Loss')
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(path_plot + 'Loss_' +str(nama_plot) + '.png')
    plt.clf()

def check_model(path, num_mode):
        path_mod = path + 'model/' + str(num_mode) + '.pth'
        checkpoint = torch.load(path_mod)
        try:
            print('epoch: {} \nAcc: {} \nloss: {} \ntransform: {} \nbatch sizes: {} \npretrain: {} \nusing sampler: {} '
                  '\ndropout: {} \nweight_decay: {} \n Data: {}'.format(
                checkpoint['epoch'],
                checkpoint['acc'],
                checkpoint['loss'],
                checkpoint['transform'],
                checkpoint['batch_sizes'],
                checkpoint['pretrain'],
                checkpoint['using_sampler'],
                checkpoint['dropout'],
                checkpoint['decay'],
                checkpoint['slice'],
            ))
        except:
            print('not found')
