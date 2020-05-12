import torch
from torch.optim import Adam
import torch.nn as nn

from KD_Lib.models.resnet import resnet_book
from KD_Lib.TAKD.training import train_model, train_distill_model
from KD_Lib.TAKD.data_loader import get_cifar10


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Working on: ", device)

    train_loader, test_loader = get_cifar10()

    param = [32, 32, 64, 64, 128]
    teacher = resnet_book['101'](param).to(device)
    assitant1 = resnet_book['50'](param).to(device)
    assitant2 = resnet_book['34'](param).to(device)

    param2 = [16, 32, 32, 16, 8]
    student = resnet_book['18'](param2).to(device)

    loss = nn.CrossEntropyLoss()

    print("Teacher Model   : Resnet101")
    optimizerTeacher = Adam(teacher.parameters())
    teacher_results = train_model(teacher, optimizerTeacher, loss,
                                  train_loader, test_loader, device,
                                  'teacher.pth.tar')
    print(teacher_results)

    print("Assistant Models: Resnet50")
    optimizerTA1 = Adam(assitant1.parameters())
    TA1_results = train_distill_model([teacher], assitant1, optimizerTA1, loss,
                                      3, train_loader, test_loader, device,
                                      0.8, 'assistant1.pth.tar')
    print(TA1_results)

    print("Assistant Models: Resnet50 and Resnet34")
    optimizerTA2 = Adam(assitant2.parameters())
    TA2_results = train_distill_model([teacher, assitant1], assitant2,
                                      optimizerTA2, loss, 3, train_loader,
                                      test_loader, device, 0.8,
                                      'assistant2_pth.tar')
    print(TA2_results)

    print("Student         : Resnet18")
    optimizerStudent = Adam(student.parameters())
    student_results = train_distill_model([assitant2, assitant1], student,
                                          optimizerStudent, loss, 3,
                                          train_loader, test_loader, device,
                                          0.8, 'assistant2_pth.tar')
    print(student_results)


if __name__ == '__main__':
    main()
