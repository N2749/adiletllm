import configparser


def get(section: str, field: str):
    config = configparser.ConfigParser()
    config_path = "config.ini"
    config.read(config_path)
    print(config.sections())
    return config[section][field]
