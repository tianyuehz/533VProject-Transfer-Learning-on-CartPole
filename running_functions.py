import gym
import torch
import numpy as np

device = ('cuda' if torch.cuda.is_available() else 'cpu')
true_params = torch.Tensor([9.8, 1.0, 0.1])
std = torch.Tensor([3, 0.5, 0.05])

def de_normalize(pred_mu):
    '''For Moving from -1 and 1 output to continous values - Inference'''
    return (std[:pred_mu.shape[1]] * pred_mu) + true_params[:pred_mu.shape[1]]

def normalize(gt_mu):
    '''Moving GT values to -1 and 1 output - Training'''
    return (gt_mu - true_params[:gt_mu.shape[1]])/std[:gt_mu.shape[1]]

def fit(model, data_x, target_x, batch_size, optimizer, criterion):
    model.train()
    running_loss = 0.0
    n = data_x.shape[0]

    #for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
    for i in range(0, n, batch_size):
        batch_data, mu = data_x[i:i+batch_size], target_x[i:i+batch_size]

        #mu = mu *np.random.rand()

        '''You want to normalize the input data?'''
        #print("batch_data", batch_data.shape)
        # temp = []
        # for i in range(3):
        #     window = batch_data.view(-1)[i*5:(i+1)*5]
        #     obs = window[:4].numpy()
        #     action = window[4]
        #     #print("obs", obs)
        #     normed = obs_normalizer(obs) 
        #     #print("normed", normed);stop
        #     temp.extend(normed)
        #     temp.append(action)
        
        # batch_data = torch.Tensor(temp)
        batch_data, mu = batch_data.to(device), mu.to(device);
        #print(f"batch_data {batch_data[0,0:5]}")
        #print(f"mu {mu}"); #stop

        
        optimizer.zero_grad()
        mu_hat = model(batch_data)
        #mu_norm = normalize(mu)
        #print("batch_data, mu_hat, mu", batch_data.shape, mu_hat.shape, mu.shape)
        #print("mu_hat, mu_norm",mu_hat[0], mu[0])
        loss = criterion(mu_hat, mu); #print("loss per batch", loss)
        running_loss += loss.item()
        
        loss.backward()
        optimizer.step()

        # for name, param in model.named_parameters():
        #     print(name, param.grad)
        # stop

    train_loss = running_loss/n
    return train_loss

def validate(model, data_x, target_x, batch_size, criterion):
    model.eval()
    #running_loss= [0.0, 0.0, 0.0]
    running_loss = 0.0
    n = data_x.shape[0]

    with torch.no_grad():
        #for i, data in tqdm(enumerate(dataloader), total=int(len(val_data)/dataloader.batch_size)):
        for i in range(0, n, batch_size):
            batch_data, mu = data_x[i:i+batch_size], target_x[i:i+batch_size]
            batch_data, mu = batch_data.to(device), mu.to(device) #print("batch_data", batch_data.shape);stop

            mu_hat = model(batch_data)
            #mu_hat_cont = de_normalize(mu_hat)

            #losses
            '''Do you know which contributes more?'''
            # for i in range(mu.shape[1]):
            #     loss = criterion(mu_hat_cont[:,i], mu[:,i])
            #     running_loss[i] += loss.item()
            
            #print("mu_hat, mu_norm",mu_hat[0], mu[0])
            loss = criterion(mu_hat, mu)
            running_loss += loss.item()
    
    # val_loss_g = running_loss[0]/n
    # val_loss_mc = running_loss[1]/n
    #val_loss_mp = running_loss[2]/n

    val_loss = running_loss/n
    return val_loss



def obs_normalizer(obs):
    """Normalize the observations between 0 and 1
    
    If the observation has extremely large bounds, then clip to a reasonable range before normalizing; 
    (-2,2) should work.  (It is ok if the solution is specific to CartPole)
    
    Args:
        obs (np.ndarray): shape (4,) containing an observation from CartPole using the bound of the env
    Returns:
        normed (np.ndarray): shape (4,) where all elements are roughly uniformly mapped to the range [0, 1]
    
    """

    env = gym.make("CartPole-v0")

    # HINT: check out env.observation_space.high, env.observation_space.low
    
    # TODO: implement this function
    #raise NotImplementedError('TODO')
    
    # min-max normalization
    
    highs = np.array(env.observation_space.high, dtype=np.float128)
    lows = np.array(env.observation_space.low, dtype=np.float128)
    
    # For extremely large bounds
    min_, max_ = -2, 2
    obs[1], obs[3] = np.clip(obs[1], min_, max_), np.clip(obs[3], min_, max_)
    
    # set the new clip values
    highs[1], highs[3] = max_, max_
    lows[1], lows[3] = min_, min_
    
    # min-max normalization
    normed = (obs - lows)/(highs - lows)
    
    return normed