import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

def do_epoch(model, dataloader, criterion, optim=None, device="cpu"):
    total_loss = 0
    total_accuracy = 0
    if optim is not None:
        model.train()
    else:
        model.eval()

    for i, data in enumerate(tqdm(dataloader, leave=True)):
        x, y_true = data
        x.to(device)
        y_true.to(device)
        
        if optim is not None:
            optim.zero_grad()

        y_pred = model(x)
        
        loss = criterion(y_pred, y_true)
        
        if optim is not None:
            loss.backward()
            optim.step()

        total_loss += loss.item()
        total_accuracy += (y_pred.max(1)[1] == y_true).float().mean().item()

    mean_loss = total_loss / len(dataloader)
    mean_accuracy = total_accuracy / len(dataloader)

    return mean_loss, mean_accuracy


def run_train(model, train_loader, val_loader, test_loader, device, optim, criterion, lr_scheduler, epochs=40):
    best_accuracy = 0
    for epoch in range(epochs):
        train_loss, train_accuracy = do_epoch(model, train_loader, criterion, optim, device)

        with torch.no_grad():
            val_loss, val_accuracy = do_epoch(model, val_loader, criterion, None, device)

        tqdm.write(f'EPOCH {epoch:03d}: train_loss={train_loss:.4f}, train_accuracy={train_accuracy:.4f} '
                   f'val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}')

        if val_accuracy > best_accuracy:
            print('Saving best model...')
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pt')

        lr_scheduler.step()

    with torch.no_grad():
        fin_loss, fin_accuracy = do_epoch(model, val_loader, criterion, None, device)

    tqdm.write(f'final_val_loss={fin_loss:.4f}, final_val_accuracy={fin_accuracy:.4f}')

    print('Saving final model...')
    torch.save(model.state_dict(), 'final_model.pt')
    with torch.no_grad():
            test_loss, test_accuracy = do_epoch(model, test_loader, criterion, None, device)
            
    tqdm.write(f'test_loss={test_loss:.4f}, test_accuracy={test_accuracy:.4f}')
    
    
    
