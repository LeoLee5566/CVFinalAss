import torchvision
import os
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms
from model import *

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise Exception("cuda not available")

    img_transformer = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = torchvision.datasets.CIFAR10(root="./dataset", transform=img_transformer, train=True)
    test_set = torchvision.datasets.CIFAR10(root="./dataset", transform=torchvision.transforms.ToTensor(), train=False)

    # visualize sample data
    # writer = SummaryWriter("./runs/data_sample")
    # figure = plt.figure(figsize=(15, 15))
    # for i in range(25):
    #     img, label = train_set[i]
    #     img = img.permute(1, 2, 0)
    #     label = train_set.classes[label]
    #     plt.subplot(5, 5, i+1, title=label)
    #     plt.imshow(img)
    #
    # writer.add_figure("train_img", figure, 0)
    # writer.close()

    # load data
    train_data = DataLoader(dataset=train_set, batch_size=512, shuffle=True, num_workers=8)
    test_data = DataLoader(dataset=test_set, batch_size=512, num_workers=8)

    # create model
    dropout_ratio = 0.05
    model = Vit(img_size=32, patch_size=2, num_classes=10, num_att_layer=3, num_heads=12, cls=True, num_channel=3, drop_ratio=dropout_ratio)
    checkpoint_path = "./checkpoints/SimpleViT/epoch_10"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    # loss function
    loss_fnc = nn.CrossEntropyLoss()
    loss_fnc = loss_fnc.to(device)

    # optimizer
    learning_rate = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

    # train model
    epoch = 10
    current_epoch = 0
    total_train_step = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        current_epoch = checkpoint["epoch"]
        total_train_step = checkpoint["train_step"]

    writer = SummaryWriter("./runs/training_logs")
    for i in range(epoch):
        current_epoch += 1
        print("---------epoch {}---------".format(current_epoch))

        total_train_accuracy = 0
        for data in train_data:
            x_train, y_train = data
            x_train = x_train.to(device)
            y_train = y_train.to(device)

            y_pred = model(x_train)
            loss = loss_fnc(y_pred, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step += 1
            total_train_accuracy += (y_pred.argmax(1) == y_train).sum()
            if total_train_step % 200 == 0:
                writer.add_scalar("train_loss_vit", loss.item(), total_train_step)
        scheduler.step()
        writer.add_scalar("train_acc_vit", total_train_accuracy / len(train_set), total_train_step)
        print("train acc of epoch {}: {}".format(current_epoch, total_train_accuracy / len(train_set)))

        total_test_loss = 0
        total_test_accuracy = 0
        with torch.no_grad():
            for data_test in test_data:
                x_test, y_test = data_test
                x_test = x_test.to(device)
                y_test = y_test.to(device)

                y_test_pred = model(x_test)
                test_loss = loss_fnc(y_test_pred, y_test)

                total_test_loss += test_loss.item()
                total_test_accuracy += (y_test_pred.argmax(1) == y_test).sum()
        writer.add_scalar("test_loss_vit", total_test_loss, total_train_step)
        writer.add_scalar("test_acc_vit", total_test_accuracy / len(test_set), total_train_step)
        print("test loss of epoch {}: {}".format(current_epoch, total_test_loss))
        print("test acc of epoch {}: {}".format(current_epoch, total_test_accuracy / len(test_set)))

    writer.close()


