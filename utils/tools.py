def print_color_text(text, color):
    color_dice = {'black': '30', 'red': '31', 'green': '32', 'orange': '33',
                  'blue': '34', 'purple': '35', 'blue-green': '36', 'white': '37'}

    print("\033[{}m{}\033[0m".format(color_dice[color], text))
