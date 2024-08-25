import torch
import torchvision
from line_profiler import profile

############# code changes ###############
#import intel_extension_for_pytorch as ipex

to_device = "xpu"

@profile
def train_resnet50():
    ############# code changes ###############

    LR = 0.001
    DOWNLOAD = True
    DATA = "data/cifar10/"

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    train_dataset = torchvision.datasets.CIFAR10(
        root=DATA,
        train=True,
        transform=transform,
        download=DOWNLOAD,
    )
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128)

    model = torchvision.models.resnet50()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    model.train()
    ##################################### code changes ################################
    model = model.to(to_device)
    criterion = criterion.to(to_device)
    #model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.bfloat16)
    ##################################### code changes ################################

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        ######################### code changes #########################
        data = data.to(to_device)
        target = target.to(to_device)
        #with torch.xpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
        with torch.amp.autocast(to_device):
        ######################### code changes #########################
            output = model(data)
            loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        print(f"batch index = {batch_idx}, loss value = {loss.item()}")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        "checkpoint.pth",
    )

    print("Execution finished")

if __name__ == "__main__":
    train_resnet50()