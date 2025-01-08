import numpy as np 
import eko
from numpy import *
import yaml
from card_test import theory_card

#theorycard = '../../../theories_slim/data/theory_cards/390.yaml'
theorycard = 'ekov013/theory.yaml'

with open(theorycard,'r') as file:
    theoryy = yaml.safe_load(file)

print(isinstance(theoryy,theory_card))
