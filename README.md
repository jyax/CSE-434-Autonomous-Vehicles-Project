# ECE-CSE 434 Project Outline

## Topic: Detection of pedestrian walkways by autonomous vehicles

### Introduction

#### What is the risk and what are the potential harms?

When operating an autonomous vehicle it is crucial that during object detection the vehicle is able to interpret a multitude of signals. While stop signs and lights are crucial to prevent vehicle collisions, probably most crucial is to identify pedestrian walkways and thus pedestrians that may be crossing. Potential risks can be death or major injury depending on how fast the vehicle is moving upon impact.

Take campus for example (primarily by the engineering building) , many of the walkways are not stop sign guarded and thus an autonomous vehicle would need to take measures to see the pedestrian crossing lines and act accordingly to prevent crossing and potentially hitting someone crossing the street.

### Risk Significance

#### How significant is the risk?

In a world without wide usage of autonomous vehicles pedestrian accidents remain a major concern. In 2021 alone pedestrian fatalities in the United States alone numbered 7,388 while over 600,000 were injured, a 13% increase from 2020 ([NHTSA](https://www.nhtsa.gov/road-safety/pedestrian-safety#:~:text=In%202021%2C%207%2C388%20pedestrians%20were,tips%20to%20keep%20pedestrians%20safe.)). The risk of hitting a pedestrian can be killing/destroying whatever is crossing the walkway or at least causing major harm to them.

### Mitigations

#### What are ways the risk can be addressed by an Autonomous Vehicle?

To address this risk object detection can be utilized similar to what was done in class, this would mitigate hitting a pedestrian who is crossing. A model could be trained to identify a variety of objects that are present in walkways, however it may be more practical to identify anything that is on the walkway since collision with anything is not ideal. The model could however be trained to identify a pedestrian walkway since it is required by law to yield and thus probably slow down at pedestrian crossings.

### Approach

We plan to utilize YOLOv5 on pre-trained models for crosswalks and pedestrians. We want to have the vehicle slow down when it detects either a pedestrian or crosswalk designation. If a pedestrian is detected close to the middle of our detection field, we will gradually slow down to a stop to prevent from hitting the pedestrian.


### Results

The turtlebot will hopefully provide insight into its accuracy to detect crosswalks and the accuracy to detect pedestrians.

### Conclusion

By properly detecting crosswalks and pedestrians, we would reduce the risk of collision with pedestrians.
