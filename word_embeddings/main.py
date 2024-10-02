import torch
import torch.nn as nn
import gensim
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import pprint

good_review = """The battery life is insane. Unless you're doing very long days 
in the backcountry off-road and away from any charging opportunities, the 
solar is a complete waste. I get over 2 weeks of training at ~300mi/wk 
(road/gravel) out of one charge and even after all that the battery will 
still be at 30%, more than enough for a couple centuries. After a year or 
nearly daily use I have noticed no loss of capacity. Easy to navigate, the 
monochrome LCD data screens are perfect...no wasting energy on colors, 
graphics, or animations no one needs. The screen is massive and easy to 
see but the unit is quite thin making for a minimal profile. Even in hard 
rain it's rare to get false input on the touch screen. This puts wahoo's 
sad button-only interface and karoo's 'barely can get through one ride' 
battery life to shame."""

bad_review = """Battery life is great even without the solar charging. 
With it, more than enough for even multiple days of use. Transflective 
display is a very undervalued piece of kit. That said: What is the point 
in paying $700+ when this thing basically has near no functionality 
without being connected to your phone and to connect to the 1040, your 
phone needs multiple Garmin apps and those apps all need access to all 
of your information and place tracking cookies on your browsers that 
track you not merely on Garmin's websites and apps, but across ALL your 
web browsing activities. Down to the mouse movement and keystroke. Read
the Garmin Connect and Garmin Cookie policy on their website. So, to sum 
the product up you are paying Garmin to have them fill in a simple 
spreadsheet's worth of your cycling stats and habits that your $700 bike 
computer could easily handle offline but will of course only function if 
connected to your phone and online AND take all of your information and 
web browsing habits to be bundled and sold to data brokers and/or 
depending on your country, sent directly to your government for...safety... 
At least you can see the screen and the battery will go 2 days."""


path = "/mnt/c/Users/nicholas/Documents/GoogleNews-vectors-negative300.bin.gz"
model = KeyedVectors.load_word2vec_format(path, binary=True, limit=100000)

good_average = np.zeros(300)
good_count = 0
for token in good_review.split():
    if model.has_index_for(token):
        good_count += 1
        good_average += model[token]

good_average /= good_count

bad_average = np.zeros(300)
bad_count = 0
for token in bad_review.split():
    if model.has_index_for(token):
        bad_count += 1
        bad_average += model[token]

bad_average /= bad_count

print(f'good: {np.dot(good_average, model["good"])})')
print(f'bad: {np.dot(bad_average, model["good"])})')
