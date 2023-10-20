from tqdm import tqdm
import torch

def run_iter(model, num_epoch, loss_fn, train_loader, test_loader, val_loader, acc_fn, optim, lr_func, lr_scheduler, device='cpu', model_name='', output_dir='./model_weights/', loaded_loss=None, running_ae=False):
    model = model.to(device)

    best_val_acc = 0
    lowest_val_loss = 100000000

    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    if loaded_loss is not None:
        train_loss, train_acc, test_loss, test_acc = loaded_loss
        best_val_acc = max(test_acc)
        lowest_val_loss = min(test_loss)

    optim.zero_grad()
    for e in tqdm(range(num_epoch)):
        print(f"Epoch {e + 1}")
        train_results = train_step(model, optim, loss_fn, train_loader, acc_fn, device, running_ae)
        if acc_fn is not None:
            train_batch_losses, train_batch_accs = train_results
        else:
            train_batch_losses, train_batch_accs = train_results, [0]

        # train_batch_losses, train_batch_accs = train_step(model, optim, loss_fn, train_loader, acc_fn, device, running_ae)
        avg_train_loss = sum(train_batch_losses) / len(train_batch_losses)
        avg_train_acc = sum(train_batch_accs) / len(train_batch_accs)

        lr_scheduler.step()

        test_results = test_step(model, loss_fn, test_loader, acc_fn, device, running_ae)
        if acc_fn is not None:
            test_batch_losses, test_batch_accs = test_results
        else:
            test_batch_losses, test_batch_accs = test_results, [e]

        # test_batch_losses, test_batch_accs = test_step(model, loss_fn, test_loader, acc_fn, device, running_ae)
        avg_test_loss = sum(test_batch_losses) / len(test_batch_losses)
        avg_test_acc = sum(test_batch_accs) / len(test_batch_accs)

        train_loss.append(avg_train_loss)
        train_acc.append(avg_train_acc)
        test_loss.append(avg_test_loss)
        test_acc.append(avg_test_acc)

        print(f"Epoch {e+1} Train Loss: {avg_train_loss} Train Accuracy: {avg_train_acc} Test Loss: {avg_test_loss} Test Accuracy: {avg_test_acc}")

        if acc_fn is not None and avg_test_acc > best_val_acc:
            best_val_acc = avg_test_acc
            print(f'Saving best model with acc {best_val_acc} at {e} epoch!')
            torch.save(model, f"{output_dir + model_name}.pt")
            torch.save(optim.state_dict(), f"{output_dir + model_name}_optimizer.pt")
            torch.save(lr_scheduler.state_dict(), f"{output_dir + model_name}_lrscheduler.pt")
            # Save acc as well
            torch.save((train_loss, train_acc, test_loss, test_acc), f"{output_dir + model_name}.pickle")

        elif acc_fn is None and avg_test_loss < lowest_val_loss:
            lowest_val_loss = avg_test_loss
            print(f'Saving best model with loss {lowest_val_loss} at {e} epoch!')
            torch.save(model, f"{output_dir + model_name}.pt")
            torch.save(optim.state_dict(), f"{output_dir + model_name}_optimizer.pt")
            torch.save(lr_scheduler.state_dict(), f"{output_dir + model_name}_lrscheduler.pt")
            # Save acc as well
            torch.save((train_loss, train_acc, test_loss, test_acc), f"{output_dir + model_name}.pickle")



def train_step(model, opt, loss_fn, train_loader, acc_fn=None, device='cpu', running_ae=False):
    model.train()
    epoch_loss = []
    epoch_acc = []

    for data in train_loader:
        if len(data) == 2:
            img, label = data
            img = img.to(device)
            label = label.to(device)
        elif len(data) == 1:
            img = data[0]
            img = img.to(device)
            label = img
        elif len(data) == 3:
            img, mask, label = data
        else:
            img = data.to(device)
            label = img

        if running_ae:
            label = img

        opt.zero_grad()
        y_pred = model(img)
        loss = loss_fn(y_pred, label)
        if acc_fn is not None:
            acc = acc_fn(y_pred, label)
            epoch_acc.append(acc.item())

        loss.backward()
        opt.step()

        epoch_loss.append(loss.item())

    if acc_fn is not None:
        return epoch_loss, epoch_acc

    else:
        return epoch_loss


@torch.no_grad()
def test_step(model, loss_fn, test_loader, acc_fn, device='cpu', running_ae=False):
    model.eval()
    epoch_loss = []
    epoch_acc = []

    for data in test_loader:
        if len(data) == 2:
            img, label = data
            img = img.to(device)
            label = label.to(device)
        elif len(data) == 1:
            img = data[0]
            img = img.to(device)
            label = img
        elif len(data) == 3:
            img, mask, label = data
        else:
            img = data.to(device)
            label = img

        if running_ae:
            label = img

        y_pred = model(img)
        loss = loss_fn(y_pred, label)
        epoch_loss.append(loss.item())
        if acc_fn is not None:
            acc = acc_fn(y_pred, label)
            epoch_acc.append(acc.item())

    if acc_fn is not None:
        return epoch_loss, epoch_acc

    else:
        return epoch_loss
