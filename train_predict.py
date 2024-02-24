import model_ensemble
import prepare_data
import tokenizer_data

train, test, sub = prepare_data.prepare_data()

tf_train, tf_test = tokenizer_data.tokenizer_data(train, test)

y_train = train['label'].values

model = model_ensemble.get_model()
print(model)

if len(test.text.values) <= 5:
    # if not, just sample submission
    sub.to_csv('submission.csv', index=False)
else:
    model.fit(tf_train, y_train)
    final_preds = model.predict_proba(tf_test)[:,1]
    sub['generated'] = final_preds
    sub.to_csv('submission.csv', index=False)