import fortune_tell


path_to_fortunes = r"fortune_bank.txt"
with open(path_to_fortunes, "r", encoding="utf8") as f:
    fortunes = f.read()
print(str(len(fortunes)) + " character(s)")
chars = set(fortunes)
print("'~' is a good pad character: ", "~" not in chars)

fortunes = fortunes.split("\n")


lm3 = fortune_tell.train_lm(fortunes, 9)


def tell_fortune():
    return fortune_tell.generate_text(lm3, 9)


