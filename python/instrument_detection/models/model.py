"""
Model definitions: a plain VGG classifier, a ResNet classifier and the VGG multi-head model with one head per instrument family.
All of them take an HCQT patch (harmonics, frames, bins) and return one logit per instrument per frame.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from instrument_detection.config import ACTIVE_MODEL_NAME, NUM_INSTRUMENTS, N_HARMONICS
from instrument_detection.models.models_config import MODEL_BLUEPRINTS

class VGGBaseModel(nn.Module):
    """Single-head VGG model: six convolutions, a frequency aggregator and one 1x1 classifier over all instruments."""
    def __init__(self, channels, dropout_rate, final_freq_width,
                 input_channels=N_HARMONICS, num_instruments=NUM_INSTRUMENTS):
        """Builds the six-convolution backbone, the aggregator and the classifier.
        channels gives the width of each pair of convolutions, dropout_rate is applied before the classifier.
        """
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
        """Maps an HCQT patch to logits of shape (batch, instruments, frames)."""
        x = self.backbone(x)
        x = self.freq_aggregator(x)
        x = torch.squeeze(x, dim=3)
        x = self.dropout(x)

        return self.final_classifier(x)

    def initialize_weights(self):
        """Sets Kaiming normal weights on every convolution, and 1 and 0 for the batch norm scale and shift."""
        for m in self.modules():

            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def count_parameters(self):
        """Returns the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class ResNetBlock(nn.Module):
    """
    Basic two-convolution residual block.
    The shortcut gets a 1x1 convolution when the stride or the channel count changes.
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=(1, 1)):
        """Builds the two convolutions and, if the shape changes, the 1x1 shortcut."""
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
        """Adds the shortcut to the two-convolution path, then applies ReLU."""
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)

        return F.relu(out, inplace=True)

class ResNetBase(nn.Module):
    """Single-head ResNet with four stages, a frequency aggregator and a 1x1 classifier over all instruments."""
    def __init__(self, channels, num_blocks, dropout_rate, final_freq_width, input_channels=N_HARMONICS, num_instruments=NUM_INSTRUMENTS):
        """Builds the stem, the four residual stages, the aggregator and the classifier.
        num_blocks gives the number of blocks in each stage.
        """
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
        """
        Builds one ResNet stage.
        Only the first block uses the given stride, the rest keep the size.
        Updates self.in_channels so the next stage knows its input width.
        """
        strides = [stride] + [(1, 1)] * (num_blocks - 1)
        layers = []

        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        """Maps an HCQT patch to logits of shape (batch, instruments, frames)."""
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
        """Sets Kaiming normal weights on every convolution, and 1 and 0 for the batch norm scale and shift."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def count_parameters(self):
        """Returns the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class VGGMultiHead(nn.Module):
    """
    Shared VGG backbone with four heads (piano, strings, woodwinds, brass).
    Each head has its own aggregator, dropout and classifier.
    The head outputs are concatenated in that order, giving 11 outputs per frame.
    """
    def __init__(self, channels, dropout_rates, final_freq_width,
                 input_channels=N_HARMONICS, num_instruments=NUM_INSTRUMENTS):
        """Builds the input norm, the shared backbone and the four heads.
        dropout_rates is a dict with one rate per family, so each head can be regularised differently.
        """
        super(VGGMultiHead, self).__init__()

        # Normalises each harmonic channel per example, so loudness differences between recordings matter less.
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

        # Number of instruments in each family.
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
        """Builds one head's frequency aggregator: a conv over the whole frequency axis, batch norm and LeakyReLU."""
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, final_freq_width), bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.01, inplace=True)
        )

    def forward(self, x):
        """Maps an HCQT patch to logits of shape (batch, 11, frames), ordered piano, strings, woodwinds, brass."""
        # The backbone runs once, then every head works on the same features.
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

        # Order matches the instrument order in config.
        return torch.cat([piano_out, strings_out, woodwinds_out, brass_out], dim=1)

    def initialize_weights(self):
        """Sets Kaiming normal weights on every convolution, and 1 and 0 for the batch norm scale and shift."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def count_parameters(self):
        """Returns the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def build_model():
    """
    Builds the model named by ACTIVE_MODEL_NAME from its blueprint.
    Fills in the instrument count, input channels and final frequency width, which are fixed for this project.
    Prints the parameter count.
    """
    print(f"Building Model: '{ACTIVE_MODEL_NAME}'")

    blueprint = MODEL_BLUEPRINTS[ACTIVE_MODEL_NAME]
    model_class_name = blueprint['model_class']
    ModelClass = globals()[model_class_name]

    model_params = blueprint.get('params', {})
    model_params['num_instruments'] = NUM_INSTRUMENTS
    model_params['input_channels'] = N_HARMONICS
    # 11 is the width left on the frequency axis after the three stride-2 convolutions.
    model_params['final_freq_width'] = 11

    model = ModelClass(**model_params)

    print(f"Successfully built '{model_class_name}' with {model.count_parameters():,} parameters.")

    return model

def build_model_with_params(model_params):
    """
    Same as build_model, but uses the given parameters (for example from the hyperparameter search) instead of the blueprint ones.
    The final frequency width must already be in model_params.
    """
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