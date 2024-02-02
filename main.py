from utils.notifier import Notifyer

if __name__ == "__main__":
    try:
        l = [1,2,3]
        print(l[4])
    except Exception as e:
        n = Notifyer(mode="slack")
        n.send_exception("Hi!")