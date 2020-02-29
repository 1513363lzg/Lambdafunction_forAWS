import json
import boto3
from pre_processing.pre_processing import PreProcessor
from datetime import datetime

my_pre_processor = PreProcessor(padding_size=40, max_dictionary_size=50000)
sage_maker_client = boto3.client("runtime.sagemaker")

def lambda_handler(event, context):
    request_time = datetime.now().strftime("%Y-%b-%d-%H-%M-%S")
    tweet = event["tweet"]
    prebegin_time = datetime.now()

    features = my_pre_processor.pre_process_text(tweet)

    pre_endtime = datetime.now()


    model_payload = {
        'features_input': features
    }

    model_start =  datetime.now()

    model_response = sage_maker_client.invoke_endpoint(
        EndpointName="tweets-model",
        ContentType="application/json",
        Body=json.dumps(model_payload))

    model_end = datetime.now()

    result = json.loads(model_response["Body"].read().decode())

    response = {}
    response["Date and time of the request"] = request_time
    response["tweet"] = tweet

    if result["predictions"][0][0] >= 0.5:
        response["sentiment"] = "positive"
    else:
        response["sentiment"] = "negative"

    response["Probability from the Model"] = result["predictions"][0][0]
    response["Pre_Processing_Time"] = (pre_endtime - prebegin_time).total_seconds()
    response["Model_Inference_Time"] = (model_end - model_start).total_seconds()

    s3 = boto3.client('s3')
    s3.put_object(
    Bucket="twitter-text",
    Key="tweet_sentresult.json", 
    Body=str(json.dumps(response, indent=2)))

    # TODO implement
    return response
