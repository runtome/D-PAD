import numpy as np

def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true):
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0) 
    d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    return (u/d).mean()

# def Corr(pred, true):
#     sig_p = np.std(pred, axis=0)
#     sig_g = np.std(true, axis=0)
#     m_p = pred.mean(0)
#     m_g = true.mean(0)
#     ind = (sig_g != 0)
#     corr = ((pred - m_p) * (true - m_g)).mean(0) / (sig_p * sig_g)
#     corr = (corr[ind]).mean()
#     return corr
def Corr(pred, true):
    sig_p = np.std(pred, axis=0)
    sig_g = np.std(true, axis=0)
    m_p = pred.mean(0)
    m_g = true.mean(0)
    
    # Check dimensions
    print("sig_p shape:", sig_p.shape)
    print("sig_g shape:", sig_g.shape)
    print("m_p shape:", m_p.shape)
    print("m_g shape:", m_g.shape)
    
    ind = (sig_g != 0)
    corr = ((pred - m_p) * (true - m_g)).mean(0) / (sig_p * sig_g)
    
    # Ensure the index matches the shape
    if corr.ndim > 1:
        corr = corr[ind].mean()
    else:
        corr = corr.mean()  # Handle cases where corr is 1D
    
    return corr


def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    return np.mean((pred-true)**2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def metric_(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    #corr1 = CORR(pred, true)
    # corr = Corr(pred, true)
    return mae,mse,rmse,mape,mspe#,corr

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    return mae,mse
