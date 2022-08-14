import argparse

from ..utils import read_sense_making_taskA
from sklearn import metrics
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer,TrainingArguments,EarlyStoppingCallback
# from models.T5ForWordPairModel import T5ForWordPairClassification



def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    prediction_labels = predictions.argmax(-1)
    acc = metrics.accuracy_score(labels,prediction_labels)
    return {'acc':acc}


def parser_args():
    parser = argparse.ArgumentParser(description='Train Model on Dataset.')
    parser.add_argument('--model', help='the path of the model')
    args = parser.parse_args()

    return args

def tokenize_function(examples,max_length=256):
    tokenized_sentence = tokenizer(examples['sent0'],examples['sent1'],padding='max_length',max_length = max_length,truncation = True)
    return tokenized_sentence




if __name__ == '__main__':
    args = parser_args()

    dataset_path = "SenseMaking"
    train_set,dev_set = read_sense_making_taskA(dataset_path=dataset_path)
    output_path = "ALL_RESULTS\\SM_"+args.model
    tokenizer = BertTokenizer.from_pretrained(args.model, use_fast=True)
        
    tokenized_train_dataset = train_set.map(
        tokenize_function, 
        batched=False, 
        remove_columns=['sent0','sent1'], 
        #batch_size=3000
    )
    tokenized_val_dataset = dev_set.map(
        tokenize_function, 
        batched=False, 
        remove_columns=['sent0','sent1'], 
        #batch_size=3000
    )

    model = BertForSequenceClassification.from_pretrained(args.model).cuda()
    
    if "base" in args.model:
        training_args = TrainingArguments(
            output_dir            = output_path,
            evaluation_strategy   = "steps",
            eval_steps            = 50,
            learning_rate         = 1e-5,
            weight_decay          = 5e-7,
            save_steps            = 35000,
            num_train_epochs      = 10,
            remove_unused_columns = True,
            warmup_steps          = 100,
            logging_strategy      = "no",
            per_device_train_batch_size = 32,
            per_device_eval_batch_size  = 32,
            lr_scheduler_type     = "cosine",
            #label_smoothing_factor= 0.05,
            metric_for_best_model = "eval_loss",
            save_total_limit       = 1,
            load_best_model_at_end = True,
        )
    else:
        training_args = TrainingArguments(
            output_dir            = output_path,
            evaluation_strategy   = "steps",
            eval_steps            = 50,
            learning_rate         = 1e-5,
            weight_decay          = 5e-7,
            save_steps            = 35000,
            num_train_epochs      = 10,
            remove_unused_columns = True,
            warmup_steps          = 100,
            logging_strategy      = "no",
            per_device_train_batch_size = 16,
            per_device_eval_batch_size  = 16,
            lr_scheduler_type     = "cosine",
            #label_smoothing_factor= 0.05,
            metric_for_best_model = "eval_loss",
            save_total_limit       = 1,
            load_best_model_at_end = True,
        )
    early_stopping = EarlyStoppingCallback(early_stopping_patience=5)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        #data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping],
    )
    trainer.train()

    max_acc = max([i['eval_acc'] for i in trainer.state.log_history if 'eval_acc' in i.keys()])

    print("max_acc:"+str(max_acc))

    with open('ALL_RESULTS\\SM_'+args.model+'_8_12.txt','a+') as f:
        f.writelines([str(i)+'\n' for i in [max_acc]])
    # save best model
    with open('ALL_RESULTS\\SM_'+args.model+'_8_12.txt', 'r') as f:
        data = f.readlines()
        numbers_float = [float(line) for line in data]
        if numbers_float.index(max(numbers_float)) == len(numbers_float)-1:
            trainer.save_model()
    print("##########################!!DONE!!################################")