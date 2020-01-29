
from termcolor import colored


""" prints progress in colored format """
def progress(epochs, epoch, loss, val_acc, elapsed):
    epochs = colored(epoch, "cyan", attrs=['bold']) + colored("/", "cyan", attrs=['bold']) + colored(epochs, "cyan", attrs=['bold'])
    loss = colored(round(loss, 6), "cyan", attrs=['bold'])
    val_acc = colored(round(val_acc, 4), "cyan", attrs=['bold']) + colored(" %", "cyan", attrs=['bold'])
    time = colored(round(elapsed, 2), "cyan", attrs=['bold']) + colored(" sec", "cyan", attrs=['bold'])

    print(" ")
    print("\nepoch {} - loss: {} - val_acc: {} - elapsed: {}".format(epochs, loss, val_acc, time))
    print("\n___________________________________________________________________________\n")