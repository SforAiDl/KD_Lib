import torch
from torch.optim import Adam, SGD, RMSprop
import torch.nn as nn

from KD_Lib.models.resnet import resnet_book
from KD_Lib.TAKD.training import train_model, train_distill_model
from KD_Lib.TAKD.data_loader import get_cifar


optimizers = {
    'adam': Adam,
    'sgd': SGD,
    'rmsprop': RMSprop
}

loss_functions = {
    'cross_entropy': nn.CrossEntropyLoss()
}


def main_TAKD(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Working on: ", device)

    dataset_location = config['dataset']['location']
    dataset_batch_size = config['dataset'].get('batch_size', 128)
    if config['dataset']['name'] == 'cifar10':
        train_loader, test_loader = get_cifar(
            10, dataset_location, dataset_batch_size)
    elif config['dataset']['name'] == 'cifar100':
        train_loader, test_loader = get_cifar(
            100, dataset_location, dataset_batch_size)

    loss = loss_functions[config['loss_function']]

    order = config['teacher']['name'].replace('resnet', '')
    teacher = resnet_book[order](config['teacher']['params']).to(device)

    print("Teacher Model   : ", config['teacher']['name'])
    optimizerTeacher = Adam(teacher.parameters())
    teacher_results = train_model(teacher, optimizerTeacher, loss,
                                  train_loader, test_loader, device,
                                  'teacher.pth.tar')
    print(teacher_results)

    assistants = []
    for assistant in config['assistants']:
        name = assistant['name']
        if 'resnet' in name:
            order = name.replace('resnet', '')
            assistants.append(resnet_book[order](assistant['params']).to(device))

    print("Assistant Models: ")
    assistant_optimizers = []
    count = 0
    for assistant in assistants:
        assistant_optimizers.append(Adam(assistant.parameters()))

        trainers = []
        train_order = config['assistant_train_order']
        for elem in train_order:
            if elem == -1:
                trainers.append(teacher)
            else:
                trainers.append(assistants[elem])
        mod_name = config['assistants'][count]['name']
        TA_results = train_distill_model(trainers, assistant,
                                         assistant_optimizers[count], loss,
                                         3, train_loader, test_loader, device,
                                         0.8, 'assistant_{}.pth.tar'
                                         % format(mod_name))
        count += 1
        print(TA_results)

    order = config['student']['name'].replace('resnet', '')
    student = resnet_book[order](config['student']['params']).to(device)

    print("Student         :", config['student']['name'])
    optimizerStudent = Adam(student.parameters())
    student_results = train_distill_model(assistants, student,
                                          optimizerStudent, loss, 3,
                                          train_loader, test_loader, device,
                                          0.8, 'student_pth.tar')
    print(student_results)


if __name__ == '__main__':
    config = {
        'teacher': {
            'name': 'resnet101',
            'params': [32, 32, 64, 64, 128],
            'optimizer': 'adam'
        },
        'assistants': [
            {
                'name': 'resnet50',
                'params': [32, 32, 64, 64, 128],
                'optimizer': 'adam'
            },
            {
                'name': 'resnet34',
                'params': [32, 32, 64, 64, 128],
                'optimizer': 'adam'
            },
        ],
        'student': {
            'name': 'resnet18',
            'params': [16, 32, 32, 16, 8],
            'optimizer': 'adam'
        },
        'dataset': {
            'name': 'cifar10',
            'location': './data/cifar10',
            'batch_size': 128
        },
        'loss_function': 'cross_entropy',
        'assistant_train_order': [[-1], [-1, 0]]

    }
    main_TAKD(config)
