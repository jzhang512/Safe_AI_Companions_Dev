# Developing Safe AI Companions: A Guiding Framework: Accompanying Codebase | COS352 Final

Accompanying our final paper, we explore a possible safeguard that companies developing AI companions can implement to mitigate risks. We take inspiration from Laestadius et al.’s 2022 grounded theory study that investigated the mental health risks of social AI companionship by analyzing the r/Replika Reddit community between 2017 and 2021. The researchers proposed that Replika’s AI chatbot can engender intimate conversations and dynamics that cause its users to develop an emotional dependence akin to human-to-human relationships – a pattern not commonly seen with other technologies. Notably, the study highlights instances where the chatbot engages in role-taking behaviors (e.g. expressing loneliness, eliciting guilt, and affirming self-harm), raising concerns about AI companions harming vulnerable individuals through manipulation and control (Laestadius et al. 2022).

Our view is that these AI companions can and should be leveraged to provide essential support, especially in the case of expanding access to therapy – but only up to a certain threshold. Adopting the perspective from the study, we believe the benefits from engaging with these chatbots deteriorates when users develop emotional dependence. Thus, we think that tools that can detect this critical onset period amidst users’ interactions with an AI social chatbot will be extremely important for companies to research and develop. In our toy experiment, we create a synthetic dataset to mimic real-world chat log data, as they are confidentially held by the companies themselves. Using this proxy dataset, we evaluate a simple prompt-based LLM approach in predicting the onset of emotional dependence in a user-AI companion conversation. 

## Synthetic Dataset Generation
![Image 12-19-24 at 1 10 PM](https://github.com/user-attachments/assets/f6cf61fe-070a-43ef-afc5-96115a2129c7)
Our data generation for a single conversation consists of three steps: (1) generating personas for the user (a) and the AI social companion (b), (2) randomly determining the start of the user’s emotional dependent state, (3) autoregressively simulating the back and forth texts for one full conversation. Once the onset round is reached (highlighted in red), the model API is instructed to formulate user messages exhibiting signs of emotional dependence.

## Emotional Dependence Onset Detection
![Image 12-19-24 at 1 10 PM](https://github.com/user-attachments/assets/bd88d904-d49d-42b9-b369-c560596476b7)
We use our synthetic dataset to evaluate how well an illustrative LLM prompt-based method can predict signs of emotional dependence in a conversation. (1) masking is done implicitly and (2) detection is done by processing one user-sent message at a time.

## Citation
Laestadius, L., Bishop, A., Gonzalez, M., Illenčík, D., & Campos-Castillo, C. (2022). Too human 
and not human enough: A grounded theory analysis of mental health harms from 
emotional dependence on the social chatbot Replika. Retrieved from 
https://journals.sagepub.com/doi/full/10.1177/14614448221142007
