import sys

def check_key_exits_with_pvkeys(previous_keys_list, key, config_data):
    if key not in config_data:
        previous_keys = '->'.join(previous_keys_list)
        print(f'under {previous_keys}, {key} and its value should be added in config file!!!')
        sys.exit(1)

def check_key_exits(key, config_data):
    if key not in config_data:
        print(f'{key} and its value should be added in config file!!!')
        sys.exit(1)

def check_config_file(config_data):
    print('Check config file format...')

    check_key_exits('train dates', config_data)
    check_key_exits_with_pvkeys(['train dates'], 'start', config_data['train dates'])
    check_key_exits_with_pvkeys(['train dates'], 'end', config_data['train dates'])
    check_key_exits('test dates', config_data)
    check_key_exits_with_pvkeys(['test dates'], 'start', config_data['test dates'])
    check_key_exits_with_pvkeys(['test dates'], 'end', config_data['test dates'])
    check_key_exits('trade dates', config_data)
    check_key_exits_with_pvkeys(['trade dates'], 'start', config_data['trade dates'])
    check_key_exits_with_pvkeys(['trade dates'], 'end', config_data['trade dates'])

    check_key_exits('model', config_data)
    check_key_exits('target name', config_data)
    check_key_exits('futures name', config_data)
    check_key_exits('futures rule', config_data)
    check_key_exits_with_pvkeys(['futures rule'], 'fixed months', config_data['futures rule'])
    check_key_exits_with_pvkeys(['futures rule'], 'total front month', config_data['futures rule'])
    check_key_exits_with_pvkeys(['futures rule'], 'delivery week', config_data['futures rule'])
    check_key_exits_with_pvkeys(['futures rule'], 'delivery weekday', config_data['futures rule'])
    check_key_exits('predict range', config_data)

    check_key_exits('features', config_data)

    check_key_exits('hyperparameters', config_data)
    ML_model = ['LR', 'XGB']
    DL_model = ['LSTM', 'TCN', 'CRNN']
    model = config_data['model']
    if model in ML_model:
        pass
    elif model in DL_model:
        check_key_exits_with_pvkeys(['hyperparameters'], 'lookbacks', config_data['hyperparameters'])
    else:
        print(f'We do not support {model} model!!!')
        sys.exit(1)
    
    check_key_exits('preprocessing', config_data)
    check_key_exits_with_pvkeys(['preprocessing'], 'correlations', config_data['preprocessing'])
    check_key_exits_with_pvkeys(['preprocessing'], 'scaling', config_data['preprocessing'])

    check_key_exits('trading', config_data)
    check_key_exits_with_pvkeys(['trading'], 'types', config_data['trading'])
    check_key_exits_with_pvkeys(['trading'], 'stop loss', config_data['trading'])
    check_key_exits_with_pvkeys(['trading'], 'tolerance', config_data['trading'])

    print('Config file format is correct!!!')