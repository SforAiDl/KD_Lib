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
    dataset_num_classes = config['dataset']['num_classes']
    dataset_num_channels = config['dataset']['num_channels']
    dataset_batch_size = config['dataset'].get('batch_size', 128)

    if 'cifar' in str(config['dataset']['name']):
        train_loader, test_loader = get_cifar(
            dataset_num_classes, dataset_location, dataset_batch_size)

    loss = loss_functions[config['loss_function']]

    order = config['teacher']['name'].replace('resnet', '')
    teacher = resnet_book[order](
        config['teacher']['params'], dataset_num_classes,
        dataset_num_channels).to(device)

    print("Teacher Model   : ", config['teacher']['name'])
    optimizerTeacher = Adam(teacher.parameters())
    epochs = config['teacher']['train_epoch']
    teacher_results = train_model(teacher, optimizerTeacher, loss,
                                  train_loader, test_loader, device,
                                  'teacher.pth.tar', epochs)
    print(teacher_results)

    assistants = []
    for assistant in config['assistants']:
        name = assistant['name']
        if 'resnet' in name:
            order = name.replace('resnet', '')
            ta_model = resnet_book[order](assistant['params'],
                                          dataset_num_classes,
                                          dataset_num_channels).to(device)
            assistants.append(ta_model)

    print("Assistant Models: ")
    assistant_optimizers = []
    count = 0
    for assistant in assistants:
        assistant_optimizers.append(Adam(assistant.parameters()))
        trainers = []
        train_order = config['assistant_train_order'][count]
        for elem in train_order:
            if elem == -1:
                trainers.append(teacher)
            else:
                trainers.append(assistants[elem])
        mod_name = config['assistants'][count]['name']

        epochs = config['assistants'][count]['train_epoch']
        TA_results = train_distill_model(trainers, assistant,
                                         assistant_optimizers[count], loss,
                                         3, train_loader, test_loader, device,
                                         0.8, 'assistant_%s.pth.tar'
                                         % format(mod_name), epochs)
        count += 1
        print(TA_results)

    order = config['student']['name'].replace('resnet', '')
    student = resnet_book[order](config['student']['params'],
                                 dataset_num_classes,
                                 dataset_num_channels).to(device)

    print("Student         :", config['student']['name'])
    epochs = config['student']['train_epoch']
    optimizerStudent = Adam(student.parameters())
    student_results = train_distill_model(assistants, student,
                                          optimizerStudent, loss, 3,
                                          train_loader, test_loader, device,
                                          0.8, 'student_pth.tar', epochs)
    print(student_results)


if __name__ == '__main__':
    config = {
        'teacher': {
            'name': 'resnet101',
            'params': [32, 32, 64, 64, 128],
            'optimizer': 'adam',
            'train_epoch': 1
        },
        'assistants': [
            {
                'name': 'resnet50',
                'params': [32, 32, 64, 64, 128],
                'optimizer': 'adam',
                'train_epoch': 1
            },
            {
                'name': 'resnet34',
                'params': [32, 32, 64, 64, 128],
                'optimizer': 'adam',
                'train_epoch': 1
            },
        ],
        'student': {
            'name': 'resnet18',
            'params': [16, 32, 32, 16, 8],
            'optimizer': 'adam',
            'train_epoch': 1
        },
        'dataset': {
            'name': 'cifar10',
            'location': './data/cifar10',
            'batch_size': 128,
            'num_classes': 10,
            'num_channels': 3
        },
        'loss_function': 'cross_entropy',
        'assistant_train_order': [[-1], [-1, 0]]
    }
    main_TAKD(config)
