import json
import re

MATCH_GET_REQUEST = r'(GET) http:\/\/localhost:8080\/(.*?)(?=HTTP)'

MATCH_POST_REQUEST = r'(POST) http:\/\/localhost:8080\/(.*?)(?=HTTP)([\s\S]*?)(?=Content-Length:(.*)\s\s(.*))'

MATCH_GET_REQUEST_AND_PAYLOAD = r'(.*?)(?=\?)*\?(.*)|(.*)'


def write_text(data, file_name):
    with open(file_name, 'w') as fp:
        for i in data:
            fp.write(i["RESOURCE"] + " " + i["PAYLOAD"] + "\n")

    return


def write_json(data, file_name):
    with open(file_name, 'w') as fp:
        fp.write(
            '[' +
            ',\n'.join(json.dumps(i) for i in data) +
            ']\n')

    return


def main():
    traffic_file = open("../../Dataset/normalTrafficTest.txt", "r")
    traffic_file2 = open("../../Dataset/anomalousTrafficTest.txt", "r")

    txt = traffic_file.read()
    txt2 = traffic_file2.read()

    txt += txt2

    requests = []

    extracted_requests = re.findall(MATCH_POST_REQUEST, txt) + re.findall(MATCH_GET_REQUEST, txt)

    for r in extracted_requests:
        if r[0] == 'GET':
            get_req_and_payload = re.findall(MATCH_GET_REQUEST_AND_PAYLOAD, r[1])[0]
            requests.append({"METHOD": "GET", "RESOURCE": get_req_and_payload[0] if get_req_and_payload[0] != "" else
            get_req_and_payload[2],
                             "PAYLOAD": get_req_and_payload[1] if get_req_and_payload[1] != "" else ""})
        elif r[0] == 'POST':
            requests.append({"METHOD": "POST", "RESOURCE": r[1], "PAYLOAD": r[4]})

    # write_json(requests, "../../Dataset/normalTrafficTest.json")
    write_text(requests, "../../Dataset/traffic_log.log")

    return


if __name__ == "__main__":
    main()
