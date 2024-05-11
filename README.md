# MARL_Environment_Chase
Code to train and evaluate MARL environment
1. Training:
   - Training script: MultiAgent_Train_Final.py
   - Arena:
       + Social: MultiAgentArena_v1d_5.py
       + Non-social : MultiAgentArena_v1d_11.py
   - Callback: callback_v1j.py to track various behavioral metrics  
   - Neural network: simple_rnn_v2_3_2.py as a simple implementation of vanilla RNN 
2. Evaluating Agents:
   - Evaluating script: MultiAgent_Evaluate_Final.py
   - Arena:
     - Evaluating in playoffs:
       + Social: MultiAgentArena_v1d_5.py
       + Non-social:
         - MultiAgentArena_v1d_11
         - MultiAgentArena_v1d_11_2: Partial vision of Partner
         - MultiAgentArena_v1d_11_3: Full vision of Partner
     - Evaluating against standard agent (that randomly sample action from a uniform distribution)
       + Social: MultiAgentArena_v1d_5_RandA1.py/MultiAgentArena_v1d_5_RandA2.py
       + Non-social: MultiAgentArena_v1d_11_RandA1.py/MultiAgentArena_v1d_11_RandA2.py
     
