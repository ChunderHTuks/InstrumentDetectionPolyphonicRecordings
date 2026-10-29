"""
Registry of the model architectures that were compared. config.py picks the active one by name.
"""
# Each entry gives the model class name (in model.py) and the arguments used to build it
MODEL_BLUEPRINTS = {
    'slim_vgg': {
        'model_class': 'VGGBaseModel',
        'params': {
            'channels': [32, 64, 128],
            'dropout_rate': 0.5
        }
    },

    'wider_vgg': {
        'model_class': 'VGGBaseModel',
        'params': {
            'channels': [64, 128, 256],
            'dropout_rate': 0.5
        }
    },

    'slim_resnet': {
        'model_class': 'ResNetBase',
        'params': {
            'channels': [16, 32, 64, 128],
            'num_blocks': [2, 2, 2, 2],
            'dropout_rate': 0.5
        }
    },

    'wider_resnet': {
        'model_class': 'ResNetBase',
        'params': {
            'channels': [32, 64, 128, 256],
            'num_blocks': [2, 2, 2, 2],
            'dropout_rate': 0.5
        }
    },

    'vgg_multihead': {
        'model_class': 'VGGMultiHead',
        'params': {
            'channels': [32, 64, 96],
            # One dropout rate per instrument family head, found by the Bayesian search
            'dropout_rates': {
                'piano': 0.14901739982187723,
                'strings': 0.34426012250150945,
                'woodwinds': 0.17159604789749142,
                'brass': 0.40464834673404443
            }
        }
    },
}