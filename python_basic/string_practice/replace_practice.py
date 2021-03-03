#用来将粘贴来的Windows路径的反斜杠替换为斜杠
def path_format(path):
    return path.replace('\\',"/")