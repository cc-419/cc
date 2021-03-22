#### 样本不均衡处理（yanan给的）

def balance_data(X_train, y_train, y_label, samples_percent_negative = 0.5):
    dresample = pd.concat([X_train, y_train], axis=1)

    # separate minority and majority classes
    negative = dresample[dresample[y_label[0]]==0]
    positive = dresample[dresample[y_label[0]]==1]

    from sklearn.utils import resample
    nsamples = int(len(positive)*samples_percent_negative)
    
    # upsample minority
    positive_upsampled = resample(positive,
                              replace=True, # sample with replacement
                              n_samples=nsamples, # match number in majority class
                              random_state=27) # reproducible results

    negative_downsampled = resample(negative,
                              replace=False, # sample without replacement
                              n_samples=nsamples, # match number in majority class
                              random_state=27) # reproducible results
                              
    X_resampled = pd.concat([positive_upsampled, negative_downsampled])
    X_train_resampled = X_resampled[X_train.columns]
    y_train_resampled = X_resampled[y_label[0]]
    return(X_train_resampled, y_train_resampled)