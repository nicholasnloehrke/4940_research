import torch


def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    print(device)


if __name__ == "__main__":
    main()