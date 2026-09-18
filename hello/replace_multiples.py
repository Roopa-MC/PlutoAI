def main():
    try:
        n = int(input("Enter n: "))
    except ValueError:
        print("Please enter an integer.")
        return

    for i in range(1, n + 1):
        if i % 2 == 0:
            print("test")
        else:
            print(i)


if __name__ == "__main__":
    main()
