import torchvision
import torchvision.transforms as transforms

def get_dataset(args):

    data_dir = '/data4/jcui7/images/data/' if 'Tian-ds' not in __file__ else '/Tian-ds/jcui7/HugeData/'
    img_size = args['img_size']
    print(data_dir)
    if args['dataset'] == 'cifar10':
        if args['normalize_data']:
            transform = transforms.Compose(
                [transforms.Resize(img_size), transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            transform = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])

        ds_train = torchvision.datasets.CIFAR10(data_dir + 'cifar10', download=True, train=True, transform=transform)
        ds_val = torchvision.datasets.CIFAR10(data_dir + 'cifar10', download=True, train=False, transform=transform)
        input_shape = [3, img_size, img_size]
        return ds_train, ds_val, input_shape