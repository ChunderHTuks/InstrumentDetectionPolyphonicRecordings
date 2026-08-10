import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ACTIVE_MODEL_NAME, NUM_INSTRUMENTS, N_HARMONICS
from models_config import MODEL_BLUEPRINTS

class VGGBaseModel(nn.Module):
    def __init__(self, channels, dropout_rate, final_freq_width,
                 input_channels=N_HARMONICS, num_instruments=NUM_INSTRUMENTS):
        super(VGGBaseModel, self).__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(input_channels, channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.LeakyReLU(0.01, inplace=True),

            nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=(1, 2), padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.LeakyReLU(0.01, inplace=True),

            nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[1]),
            nn.LeakyReLU(0.01, inplace=True),

            nn.Conv2d(channels[1], channels[1], kernel_size=3, stride=(1, 2), padding=1, bias=False),
            nn.BatchNorm2d(channels[1]),
            nn.LeakyReLU(0.01, inplace=True),

            nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[2]),
            nn.LeakyReLU(0.01, inplace=True),

            nn.Conv2d(channels[2], channels[2], kernel_size=3, stride=(1, 2), padding=1, bias=False),
            nn.BatchNorm2d(channels[2]),
            nn.LeakyReLU(0.01, inplace=True),
        )

        self.freq_aggregator = nn.Sequential(
            nn.Conv2d(channels[2], channels[2], kernel_size=(1, final_freq_width), bias=False),
            nn.BatchNorm2d(channels[2]),
            nn.LeakyReLU(0.01, inplace=True)
        )

        self.final_classifier = nn.Conv1d(channels[2], num_instruments, kernel_size=1)
        self.dropout = nn.Dropout(p=dropout_rate)

        self.initialize_weights()

        self.final_classifier = nn.Conv1d(channels[2], num_instruments, kernel_size=1)
        self.dropout = nn.Dropout(p=dropout_rate)

        self.initialize_weights()

    def forward(self, x):
        x = self.backbone(x)
        x = self.freq_aggregator(x)
        x = torch.squeeze(x, dim=3)
        x = self.dropout(x)

        return self.final_classifier(x)

    def initialize_weights(self):
        for m in self.modules():

            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class ResNetBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=(1, 1)):
        super(ResNetBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=(1, 1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()

        if stride != (1, 1) or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)

        return F.relu(out, inplace=True)

class ResNetBase(nn.Module):
    def __init__(self, channels, num_blocks, dropout_rate, final_freq_width, input_channels=N_HARMONICS, num_instruments=NUM_INSTRUMENTS):
        super(ResNetBase, self).__init__()
        self.in_channels = channels[0]

        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, self.in_channels, kernel_size=(7, 7), stride=(1, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(self.in_channels), nn.ReLU(inplace=True)
        )
        self.layer1 = self.make_layer(ResNetBlock, channels[0], num_blocks[0], stride=(1, 1))
        self.layer2 = self.make_layer(ResNetBlock, channels[1], num_blocks[1], stride=(1, 2))
        self.layer3 = self.make_layer(ResNetBlock, channels[2], num_blocks[2], stride=(1, 2))
        self.layer4 = self.make_layer(ResNetBlock, channels[3], num_blocks[3], stride=(1, 2))

        self.head_conv1 = nn.Conv2d(channels[3], channels[3], kernel_size=1, bias=False)
        self.head_bn1 = nn.BatchNorm2d(channels[3])

        self.freq_aggregator = nn.Conv2d(channels[3], channels[3], kernel_size=(1, final_freq_width), bias=False)
        self.agg_bn = nn.BatchNorm2d(channels[3])

        self.final_classifier = nn.Conv1d(channels[3] * ResNetBlock.expansion, num_instruments, kernel_size=1)
        self.dropout = nn.Dropout(p=dropout_rate)

        self.initialize_weights()

    def make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [(1, 1)] * (num_blocks - 1)
        layers = []

        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = F.relu(self.head_bn1(self.head_conv1(out)),  inplace=True)
        out = F.relu(self.agg_bn(self.freq_aggregator(out)),  inplace=True)
        out = torch.squeeze(out, dim=3)
        out = self.dropout(out)

        return self.final_classifier(out)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class VGGMultiHead(nn.Module):
    def __init__(self, channels, dropout_rates, final_freq_width,
                 input_channels=N_HARMONICS, num_instruments=NUM_INSTRUMENTS):
        super(VGGMultiHead, self).__init__()

        self.input_norm = nn.InstanceNorm2d(input_channels, affine=True)

        self.backbone = nn.Sequential(
            nn.Conv2d(input_channels, channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.LeakyReLU(0.01, inplace=True),

            nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=(1, 2), padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.LeakyReLU(0.01, inplace=True),

            nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[1]),
            nn.LeakyReLU(0.01, inplace=True),

            nn.Conv2d(channels[1], channels[1], kernel_size=3, stride=(1, 2), padding=1, bias=False),
            nn.BatchNorm2d(channels[1]),
            nn.LeakyReLU(0.01, inplace=True),

            nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[2]),
            nn.LeakyReLU(0.01, inplace=True),

            nn.Conv2d(channels[2], channels[2], kernel_size=3, stride=(1, 2), padding=1, bias=False),
            nn.BatchNorm2d(channels[2]),
            nn.LeakyReLU(0.01, inplace=True),
        )

        head_output_sizes = {
            'piano': 1,
            'strings': 4,
            'woodwinds': 4,
            'brass': 2
        }

        self.piano_aggregator = self.make_aggregator(channels[2], final_freq_width)
        self.piano_dropout = nn.Dropout(p=dropout_rates['piano'])
        self.piano_classifier = nn.Conv1d(channels[2], head_output_sizes['piano'], kernel_size=1)

        self.strings_aggregator = self.make_aggregator(channels[2], final_freq_width)
        self.strings_dropout = nn.Dropout(p=dropout_rates['strings'])
        self.strings_classifier = nn.Conv1d(channels[2], head_output_sizes['strings'], kernel_size=1)

        self.woodwinds_aggregator = self.make_aggregator(channels[2], final_freq_width)
        self.woodwinds_dropout = nn.Dropout(p=dropout_rates['woodwinds'])
        self.woodwinds_classifier = nn.Conv1d(channels[2], head_output_sizes['woodwinds'], kernel_size=1)

        self.brass_aggregator = self.make_aggregator(channels[2], final_freq_width)
        self.brass_dropout = nn.Dropout(p=dropout_rates['brass'])
        self.brass_classifier = nn.Conv1d(channels[2], head_output_sizes['brass'], kernel_size=1)

        self.initialize_weights()

    def make_aggregator(self, in_channels, final_freq_width):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, final_freq_width), bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.01, inplace=True)
        )

    def forward(self, x):
        x = self.input_norm(x)

        shared_features = self.backbone(x)

        piano_x = self.piano_aggregator(shared_features)
        piano_x = torch.squeeze(piano_x, dim=3)
        piano_x = self.piano_dropout(piano_x)
        piano_out = self.piano_classifier(piano_x)

        strings_x = self.strings_aggregator(shared_features)
        strings_x = torch.squeeze(strings_x, dim=3)
        strings_x = self.strings_dropout(strings_x)
        strings_out = self.strings_classifier(strings_x)

        woodwinds_x = self.woodwinds_aggregator(shared_features)
        woodwinds_x = torch.squeeze(woodwinds_x, dim=3)
        woodwinds_x = self.woodwinds_dropout(woodwinds_x)
        woodwinds_out = self.woodwinds_classifier(woodwinds_x)

        brass_x = self.brass_aggregator(shared_features)
        brass_x = torch.squeeze(brass_x, dim=3)
        brass_x = self.brass_dropout(brass_x)
        brass_out = self.brass_classifier(brass_x)

        return torch.cat([piano_out, strings_out, woodwinds_out, brass_out], dim=1)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def build_model():
    print(f"Building Model: '{ACTIVE_MODEL_NAME}'")

    blueprint = MODEL_BLUEPRINTS[ACTIVE_MODEL_NAME]
    model_class_name = blueprint['model_class']
    ModelClass = globals()[model_class_name]

    model_params = blueprint.get('params', {})
    model_params['num_instruments'] = NUM_INSTRUMENTS
    model_params['input_channels'] = N_HARMONICS
    model_params['final_freq_width'] = 11

    model = ModelClass(**model_params)

    print(f"Successfully built '{model_class_name}' with {model.count_parameters():,} parameters.")

    return model

def build_model_with_params(model_params):
    print(f"Building Model: '{ACTIVE_MODEL_NAME}'")

    blueprint = MODEL_BLUEPRINTS[ACTIVE_MODEL_NAME]

    model_class_name = blueprint['model_class']
    ModelClass = globals()[model_class_name]

    final_params = model_params
    final_params['num_instruments'] = NUM_INSTRUMENTS
    final_params['input_channels'] = N_HARMONICS

    model = ModelClass(**final_params)

    print(f"Successfully built '{model_class_name}' with {model.count_parameters():,} parameters.")
    return model

if __name__ == "__main__":
    import torch
    from torchinfo import summary
    model = build_model()
    batch_size = 1
    input_channels = 1
    height = 44
    width = 84

    input_shape = (batch_size, input_channels, height, width)

    print("\n" + "="*80)
    print(f"MODEL SUMMARY FOR INPUT SHAPE: {input_shape}")
    print(f"Analyzing model: '{ACTIVE_MODEL_NAME}'")
    print("="*80)
    summary(model, input_size=input_shape)
    print("="*80)
    print("Note: FLOPs (Floating Point Operations) are approximately 2 x the Mult-Adds value.")
    print("="*80)