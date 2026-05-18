import requests
import argparse
import os

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', dest='savepath', default='./', help='Save path: ~/xxx/xxx/')
    parser.add_argument('-s', dest='start_time', default='20240101000000', help='Start time: YYYYMMDDHHmmss')
    parser.add_argument('-e', dest='end_time', default='20240102000000', help='End time: YYYYMMDDHHmmss')
    args = parser.parse_args()
    return args

def validate_response(response):
    """Validate API response and return lines if successful."""
    if response.status_code != 200:
        print(f"API request failed with status code: {response.status_code}")
        print(f"Response content: {response.text}")
        return None
    if not response.text:
        print("Empty response received from API")
        return None
    return response.text.splitlines()

def download_file(url, filename, savepath, pwd, max_retries=3):
    """Download a file with retry logic."""
    attempt = 0
    while attempt < max_retries:
        try:
            rfile = requests.get(f"{url}pwd={pwd}", timeout=6)
            rfile.raise_for_status()  # Raise exception for bad status codes
            os.makedirs(savepath, exist_ok=True)  # Ensure save directory exists
            with open(os.path.join(savepath, filename), "wb") as code:
                code.write(rfile.content)
            print(f"Successfully downloaded: {filename}")
            return True
        except requests.RequestException as e:
            attempt += 1
            print(f"Retry {attempt}/{max_retries} for {filename}: {str(e)}")
            if attempt == max_retries:
                print(f"Failed to download {filename} after {max_retries} attempts")
                return False
    return False

def main():
    args = parse_args()
    print(f"Save path: {args.savepath}, Start time: {args.start_time}, End time: {args.end_time}")

    userId = 'BCSH_SUNHAOFEI'
    pwd = 'Shf980504sci!'

    timeRange = f"[{args.start_time},{args.end_time}]"
    url = f"http://10.229.90.120/music-ws/api?serviceNodeId=NMIC_MUSIC_CMADAAS&userId={userId}&interfaceId=getRadaFileByTimeRange&dataCode=RADA_L3_MST_CREF_QC&timeRange={timeRange}&dataFormat=spacetext&pwd={pwd}"

    try:
        r = requests.get(url)
        lines = validate_response(r)
        if not lines:
            return

        print(f"Number of lines in response: {len(lines)}")
        print(f"Response content: {r.text}")  # Log full response for debugging

        for num, line in enumerate(lines):
            nPos = line.find('http://')
            print(f"Processing line {num + 1}: nPos = {nPos}")
            if nPos > 1:
                sp_list = line.split(' ')
                if len(sp_list) < 4:
                    print(f"Invalid line format: {line}")
                    continue
                filename = sp_list[0]
                url = sp_list[3]
                edit_url = url.split('timestamp')[0]
                download_file(edit_url, filename, args.savepath, pwd)
            else:
                print(f"No URL found in line: {line}")

    except requests.RequestException as e:
        print(f"Error making API request: {str(e)}")

if __name__ == "__main__":
    main()