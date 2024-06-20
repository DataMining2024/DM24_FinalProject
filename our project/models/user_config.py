

# Importing DummyModel from the models package.
# The DummyModel class is located in the dummy_model.py file inside the 'models' directory.
#from models.dummy_model import DummyModel
#from models.mymodel2 import MyModel2
#from models.example import Vicuna2
#from models.testmodel import Vicunatest
#from models.better_categories import VicunaCategory
#from models.other_model_test import Vicuna2_test
#from models.vicuna13 import Vicuna13Model
#from models.yi import Yi
#from models.t5kaist import T5
#from models.Claude import Claude
#from models.stablelm import StableLM
#from models.dare_tires import DareTires
#from models.Calme import Calme
#from models.dpomistral import DPOMistral
#from models.experiment26 import Experiment26
#from models.mistroll import Mistroll
#from models.categories_annotation import Mistroll_Cats2
from models.categories_annotation import  Dareties
#from models.mixtao import Mixtao
#from models.better_cat_dare import DTCat
#from models.FusionNet import FusionNet
#from models.Qwen import Qwen
#from models.crmodel import CRModel
#from models.math_dpo import Math_DPO
#from models.textbase import TextBase
#from models.moe_dpo import Moe_DPO
#from models.CarbonBeagle import CarbonBeagle
#test


# This line establishes an alias for the DummyModel class to be used within this script.
# Instead of directly using DummyModel everywhere in the code, we're assigning it to 'UserModel'.
# This approach allows for easier reference to your model class when evaluating your models,
#UserModel = DummyModel
#UserModel = Claude
#UserModel = Qwen

#UserModel = StableLM
#UserModel = CRModel
#UserModel = Math_DPO
#UserModel = TextBase
#UserModel = CarbonBeagle
#UserModel = Moe_DPO
UserModel = Dareties
#UserModel = Calme
#UserModel = Mistroll
#UserModel = Mistroll_Cats2
#UserModel = FusionNet
#UserModel = DTCat
#UserModel = Mixtao
#UserModel = Experiment26
#UserModel = DPOMistral
#UserModel = Yi
#UserModel = Vicuna13Model
#UserModel = VicunaCategory


# When implementing your own model please follow this pattern:
#
# from models.your_model import YourModel
#
# Replace 'your_model' with the name of your Python file containing the model class
# and 'YourModel' with the class name of your model.
#
# Finally, assign YourModel to UserModel as shown below to use it throughout your script.
#


