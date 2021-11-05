from new_main import *

path = 'new_data'
if path == 'new_data':
    path_mod = 'new/fold '

else:
    path_mod = 'old/fold '

item = 3
fold1 = path_mod + '1' + '/model/' + str(item) + '.pth'
checkpoint1 = torch.load(fold1)
acc1 = checkpoint1['acc_plot']
loss1 = checkpoint1['loss_plot']


fold2 = path_mod + '2' + '/model/' + str(item) + '.pth'
checkpoint2 = torch.load(fold2)
acc2 = checkpoint2['acc_plot']
loss2 = checkpoint2['loss_plot']

fold3 = path_mod + '3' + '/model/' + str(item) + '.pth'
checkpoint3 = torch.load(fold3)
acc3 = checkpoint3['acc_plot']
loss3 = checkpoint3['loss_plot']

fold4 = path_mod + '4' + '/model/' + str(item) + '.pth'
checkpoint4 = torch.load(fold4)
acc4 = checkpoint4['acc_plot']
loss4 = checkpoint4['loss_plot']

fold5 = path_mod + '5' + '/model/' + str(item) + '.pth'
checkpoint5 = torch.load(fold5)
acc5 = checkpoint5['acc_plot']
loss5 = checkpoint5['loss_plot']

if not os.path.isdir('new_plot_all/'):
    os.makedirs('new_plot_all/')

# train_loss = model_loss[::2]
# val_loss = model_loss[1::2]

temp1 = []
temp2 = []
temp3 = []
temp4 = []
temp5 = []

for item1 in acc1:
    temp1.append(item1.item())

for item2 in acc2:
    temp2.append(item2.item())

for item3 in acc3:
    temp3.append(item3.item())

for item4 in acc4:
    temp4.append(item4.item())

for item5 in acc5:
    temp5.append(item5.item())

train_acc1 = temp1[::2]
train_acc2 = temp2[::2]
train_acc3 = temp3[::2]
train_acc4 = temp4[::2]
train_acc5 = temp5[::2]

x = max(train_acc5) + max(train_acc4) + max(train_acc3) + max(train_acc2) + max(train_acc2)
# print(x/5)

vals_acc1 = temp1[1::2]
vals_acc2 = temp2[1::2]
vals_acc3 = temp3[1::2]
vals_acc4 = temp4[1::2]
vals_acc5 = temp5[1::2]

# Saatnya plot perbandingan
# plot of the data
fig = plt.figure()
# ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
# plot perbandingan Loss
plt.title('patch')
p1=plt.plot(train_acc1)
p2=plt.plot(train_acc2)
p3=plt.plot(train_acc3)
p4=plt.plot(train_acc4)
p5=plt.plot(train_acc5)

p6=plt.plot(vals_acc1,linestyle="--")
p7=plt.plot(vals_acc2,linestyle="--")
p8=plt.plot(vals_acc3,linestyle="--")
p9=plt.plot(vals_acc4,linestyle="--")
p10=plt.plot(vals_acc5,linestyle="--")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
# ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

# plt.legend(handles=[p1, p2,p3,p4,p5,p6,p7,p8,p9,p10], title='title', bbox_to_anchor=(1.05, 1), loc='upper left', )
# plt.show()

plt.savefig('old_plot_all/patch.png')
plt.clf()

# plt.title('Perbandingan Loss 5 Fold')
# plt.plot(train_loss, label='Train Loss')
# plt.plot(val_loss, label='Val Loss')
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()
# plt.savefig(path_plot + 'Loss_' +str(nama_plot) + '.png')
# plt.clf()
