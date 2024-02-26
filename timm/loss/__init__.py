from .asymmetric_loss import AsymmetricLossMultiLabel, AsymmetricLossSingleLabel
from .binary_cross_entropy import BinaryCrossEntropy
from .cross_entropy import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, CrossEntropy
from .expected_likelihood import ExpectedLikelihoodKernel, NonIsotropicVMF
from .jsd import JsdCrossEntropy
from .MCInfoNCE import MCInfoNCE, InfoNCE
from .HedgedInstance import HedgedInstance
from .risk_prediction import LossPrediction, LossOrderLoss
