from model import PredictionModel
from os import path
from utils import load_data, accuracy, save_model


def train(args):
    import torch
    import torch.utils.tensorboard as tb

    train_logger = None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)

    print("Creating model...")
    model = PredictionModel()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'model.th')))
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    loss = torch.nn.CrossEntropyLoss()

    print("Loading dataset")
    train_data = load_data('dataset/train')
    # valid_data = load_data('dataset/valid')

    global_step = 0
    for epoch in range(args.num_epoch):
        model.train()
        loss_vals, acc_vals, vacc_vals = [], [], []
        for img, label in train_data:
            img, label = img.to(device), label.to(device)

            logit = model(img)
            loss_val = loss(logit, label)
            acc_val = accuracy(logit, label)

            loss_vals.append(loss_val.detach().cpu().numpy())
            acc_vals.append(acc_val.detach().cpu().numpy())

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            avg_loss = sum(loss_vals) / len(loss_vals)
            avg_acc = sum(acc_vals) / len(acc_vals)

            if train_logger is not None:
                train_logger.add_scalar('accuracy', avg_acc, global_step=global_step)
                train_logger.add_scalar('loss', avg_loss, global_step=global_step)
            global_step += 1
        print('epoch: {}  loss: {:.2f}  acc: {:.2f}'.format(epoch, avg_loss, avg_acc))
    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')
    parser.add_argument('-n', '--num_epoch', type=int, default=50)
    parser.add_argument('-l', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-c', '--continue_training', action='store_true')
    args = parser.parse_args()

    train(args)
