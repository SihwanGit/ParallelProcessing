#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <random>
#include <set>
#include <algorithm>
#include <stdexcept>
#include <cctype>

using namespace std;

/*
    Live Variable Analysis Dataset Generator

    config.txt 예시:

    blockNum : 4
    variableNum : 4
    cfgType : branch
    defRate : 0.3
    useRate : 0.3
    loopRate : 0.1
    maxSucc : 2
    seed : 42
    allowOverlap : false
    allowSelfLoop : false

    cfgType에 입력 가능한 값:
    chain  : B0 -> B1 -> B2 형태의 선형 CFG 생성
    branch : 뒤쪽 블록으로 여러 successor를 갖는 분기 CFG 생성
    loop   : chain 구조에 backward edge를 추가한 loop CFG 생성
    random : chain 구조에 임의 edge를 추가한 random CFG 생성

    출력 파일은 사용자가 미리 만든 dataset 폴더 안에 자동 생성된다.

    출력 파일명 형식:
    dataset\<cfgType>_<numBlocks>b_<numVars>v_def<defRate>_use<useRate>_seed<seed>.txt

    예:
    dataset\branch_4b_4v_def0.3_use0.3_seed42.txt
*/

// 하나의 Basic Block이 가지는 DEF, USE, successor 정보를 저장하는 구조체
struct Block {
    vector<int> defVars;     // 해당 블록에서 정의되는 변수들의 인덱스 목록
    vector<int> useVars;     // 해당 블록에서 정의되기 전에 사용되는 변수들의 인덱스 목록
    vector<int> succBlocks;  // 해당 블록 다음에 실행될 수 있는 successor 블록 인덱스 목록
};

// config.txt에서 읽어온 데이터셋 생성 설정값을 저장하는 구조체
struct Config {
    int numBlocks;           // 생성할 Basic Block 개수
    int numVars;             // 생성할 변수 개수

    string cfgType;          // 생성할 CFG 구조 유형: chain, branch, loop, random 중 하나

    double defRate;          // 각 변수가 특정 블록의 DEF 집합에 포함될 확률
    double useRate;          // 각 변수가 특정 블록의 USE 집합에 포함될 확률
    double loopRate;         // loop CFG에서 backward edge를 추가할 확률

    int maxSucc;             // 한 블록이 가질 수 있는 최대 successor 개수
    unsigned int seed;       // 난수 생성을 고정하기 위한 seed 값

    bool allowOverlap;       // true이면 같은 변수가 한 블록의 USE와 DEF에 동시에 포함될 수 있음
    bool allowSelfLoop;      // true이면 자기 자신으로 가는 self-loop successor를 허용함

    string outputFile;       // 자동 생성된 출력 파일 경로
};

// 문자열 앞뒤의 공백 문자를 제거하는 함수
string trim(const string& str) {
    size_t start = 0;

    while (start < str.size() && isspace(static_cast<unsigned char>(str[start]))) {
        start++;
    }

    size_t end = str.size();

    while (end > start && isspace(static_cast<unsigned char>(str[end - 1]))) {
        end--;
    }

    return str.substr(start, end - start);
}

// 문자열을 전부 소문자로 변환하는 함수
string toLowerString(string str) {
    for (char& c : str) {
        c = static_cast<char>(tolower(static_cast<unsigned char>(c)));
    }

    return str;
}

// config.txt에서 읽은 true/false 문자열을 bool 값으로 변환하는 함수
bool parseBool(const string& value) {
    string lower = toLowerString(trim(value));

    if (lower == "true" || lower == "1" || lower == "yes") {
        return true;
    }

    if (lower == "false" || lower == "0" || lower == "no") {
        return false;
    }

    throw invalid_argument("Invalid boolean value: " + value);
}

// 파일명에 들어갈 double 값을 짧은 문자열로 변환하는 함수
string doubleToFileNameString(double value) {
    string s = to_string(value);

    while (!s.empty() && s.back() == '0') {
        s.pop_back();
    }

    if (!s.empty() && s.back() == '.') {
        s.pop_back();
    }

    return s;
}

// config 설정값을 이용해 dataset 폴더 안에 저장될 출력 파일명을 자동 생성하는 함수
string makeOutputFileName(const Config& config) {
    string defStr = doubleToFileNameString(config.defRate);
    string useStr = doubleToFileNameString(config.useRate);

    string fileName =
        config.cfgType + "_" +
        to_string(config.numBlocks) + "b_" +
        to_string(config.numVars) + "v_" +
        "def" + defStr + "_" +
        "use" + useStr + "_" +
        "seed" + to_string(config.seed) +
        ".txt";

    return "dataset\\" + fileName;
}

// config.txt 파일을 읽어 Config 구조체에 저장하는 함수
Config readConfigFile(const string& configFile) {
    Config config;

    bool hasBlockNum = false;
    bool hasVariableNum = false;
    bool hasCfgType = false;
    bool hasDefRate = false;
    bool hasUseRate = false;
    bool hasLoopRate = false;
    bool hasMaxSucc = false;
    bool hasSeed = false;
    bool hasAllowOverlap = false;
    bool hasAllowSelfLoop = false;

    ifstream fin(configFile);

    if (!fin.is_open()) {
        throw runtime_error("Failed to open config file: " + configFile);
    }

    string line;

    while (getline(fin, line)) {
        line = trim(line);

        if (line.empty()) {
            continue;
        }

        if (line[0] == '#') {
            continue;
        }

        size_t colonPos = line.find(':');

        if (colonPos == string::npos) {
            throw runtime_error("Invalid config line. ':' is missing -> " + line);
        }

        string key = trim(line.substr(0, colonPos));
        string value = trim(line.substr(colonPos + 1));

        if (key == "blockNum") {
            config.numBlocks = stoi(value);
            hasBlockNum = true;
        }
        else if (key == "variableNum") {
            config.numVars = stoi(value);
            hasVariableNum = true;
        }
        else if (key == "cfgType") {
            config.cfgType = toLowerString(trim(value));
            hasCfgType = true;
        }
        else if (key == "defRate") {
            config.defRate = stod(value);
            hasDefRate = true;
        }
        else if (key == "useRate") {
            config.useRate = stod(value);
            hasUseRate = true;
        }
        else if (key == "loopRate") {
            config.loopRate = stod(value);
            hasLoopRate = true;
        }
        else if (key == "maxSucc") {
            config.maxSucc = stoi(value);
            hasMaxSucc = true;
        }
        else if (key == "seed") {
            config.seed = static_cast<unsigned int>(stoul(value));
            hasSeed = true;
        }
        else if (key == "allowOverlap") {
            config.allowOverlap = parseBool(value);
            hasAllowOverlap = true;
        }
        else if (key == "allowSelfLoop") {
            config.allowSelfLoop = parseBool(value);
            hasAllowSelfLoop = true;
        }
        else {
            throw runtime_error("Unknown config key: " + key);
        }
    }

    fin.close();

    if (!hasBlockNum) {
        throw runtime_error("Missing required config key: blockNum");
    }

    if (!hasVariableNum) {
        throw runtime_error("Missing required config key: variableNum");
    }

    if (!hasCfgType) {
        throw runtime_error("Missing required config key: cfgType");
    }

    if (!hasDefRate) {
        throw runtime_error("Missing required config key: defRate");
    }

    if (!hasUseRate) {
        throw runtime_error("Missing required config key: useRate");
    }

    if (!hasLoopRate) {
        throw runtime_error("Missing required config key: loopRate");
    }

    if (!hasMaxSucc) {
        throw runtime_error("Missing required config key: maxSucc");
    }

    if (!hasSeed) {
        throw runtime_error("Missing required config key: seed");
    }

    if (!hasAllowOverlap) {
        throw runtime_error("Missing required config key: allowOverlap");
    }

    if (!hasAllowSelfLoop) {
        throw runtime_error("Missing required config key: allowSelfLoop");
    }

    config.outputFile = makeOutputFileName(config);

    return config;
}

// Config 값들이 올바른 범위와 형식을 갖는지 검사하는 함수
void validateConfig(const Config& config) {
    if (config.numBlocks <= 0) {
        throw invalid_argument("blockNum must be positive.");
    }

    if (config.numVars <= 0) {
        throw invalid_argument("variableNum must be positive.");
    }

    if (
        config.cfgType != "chain" &&
        config.cfgType != "branch" &&
        config.cfgType != "loop" &&
        config.cfgType != "random"
        ) {
        throw invalid_argument("cfgType must be one of: chain, branch, loop, random.");
    }

    if (config.defRate < 0.0 || config.defRate > 1.0) {
        throw invalid_argument("defRate must be between 0.0 and 1.0.");
    }

    if (config.useRate < 0.0 || config.useRate > 1.0) {
        throw invalid_argument("useRate must be between 0.0 and 1.0.");
    }

    if (config.loopRate < 0.0 || config.loopRate > 1.0) {
        throw invalid_argument("loopRate must be between 0.0 and 1.0.");
    }

    if (config.maxSucc < 0) {
        throw invalid_argument("maxSucc must be non-negative.");
    }

    if (config.outputFile.empty()) {
        throw invalid_argument("outputFile must not be empty.");
    }

    if (config.cfgType == "chain" && config.maxSucc < 1 && config.numBlocks > 1) {
        throw invalid_argument("chain CFG requires maxSucc >= 1.");
    }

    if (config.cfgType == "loop" && config.maxSucc < 1 && config.numBlocks > 1) {
        throw invalid_argument("loop CFG requires maxSucc >= 1.");
    }

    if (config.cfgType == "random" && config.maxSucc < 1 && config.numBlocks > 1) {
        throw invalid_argument("random CFG requires maxSucc >= 1.");
    }
}

// v0, v1, v2 형태의 변수 이름 목록을 생성하는 함수
vector<string> generateVariableNames(int numVars) {
    vector<string> names;

    for (int i = 0; i < numVars; ++i) {
        names.push_back("v" + to_string(i));
    }

    return names;
}

// 주어진 확률에 따라 true 또는 false를 반환하는 난수 함수
bool randomProbability(mt19937& rng, double probability) {
    uniform_real_distribution<double> dist(0.0, 1.0);

    return dist(rng) < probability;
}

// 각 Basic Block의 USE와 DEF 집합을 난수 기반으로 생성하는 함수
vector<Block> generateUseDefSets(const Config& config, mt19937& rng) {
    vector<Block> blocks(config.numBlocks);

    for (int b = 0; b < config.numBlocks; ++b) {
        set<int> defSet;
        set<int> useSet;

        for (int v = 0; v < config.numVars; ++v) {
            if (randomProbability(rng, config.defRate)) {
                defSet.insert(v);
            }
        }

        for (int v = 0; v < config.numVars; ++v) {
            if (randomProbability(rng, config.useRate)) {
                if (!config.allowOverlap && defSet.count(v)) {
                    continue;
                }

                useSet.insert(v);
            }
        }

        blocks[b].defVars.assign(defSet.begin(), defSet.end());
        blocks[b].useVars.assign(useSet.begin(), useSet.end());
    }

    return blocks;
}

// 특정 블록에 successor를 중복 없이 추가하는 함수
void addSuccessor(Block& block, int succ, int maxSucc) {
    if ((int)block.succBlocks.size() >= maxSucc) {
        return;
    }

    if (find(block.succBlocks.begin(), block.succBlocks.end(), succ) == block.succBlocks.end()) {
        block.succBlocks.push_back(succ);
    }
}

// B0 -> B1 -> B2 형태의 chain CFG를 생성하는 함수
void generateChainCFG(vector<Block>& blocks, const Config& config) {
    for (int i = 0; i < config.numBlocks - 1; ++i) {
        addSuccessor(blocks[i], i + 1, config.maxSucc);
    }
}

// 현재 블록보다 뒤쪽에 있는 블록들 중 successor를 선택해 branch CFG를 생성하는 함수
void generateBranchCFG(vector<Block>& blocks, const Config& config, mt19937& rng) {
    for (int i = 0; i < config.numBlocks - 1; ++i) {
        int remainingBlocks = config.numBlocks - i - 1;
        int upperSucc = min(config.maxSucc, remainingBlocks);

        if (upperSucc <= 0) {
            continue;
        }

        uniform_int_distribution<int> succCountDist(1, upperSucc);
        int succCount = succCountDist(rng);

        uniform_int_distribution<int> succDist(i + 1, config.numBlocks - 1);

        while ((int)blocks[i].succBlocks.size() < succCount) {
            int succ = succDist(rng);
            addSuccessor(blocks[i], succ, config.maxSucc);
        }

        sort(blocks[i].succBlocks.begin(), blocks[i].succBlocks.end());
    }
}

// chain CFG를 먼저 만든 뒤 일부 블록에 backward edge를 추가해 loop CFG를 생성하는 함수
void generateLoopCFG(vector<Block>& blocks, const Config& config, mt19937& rng) {
    generateChainCFG(blocks, config);

    for (int i = 1; i < config.numBlocks; ++i) {
        if ((int)blocks[i].succBlocks.size() >= config.maxSucc) {
            continue;
        }

        if (randomProbability(rng, config.loopRate)) {
            uniform_int_distribution<int> backDist(0, i - 1);
            int target = backDist(rng);

            if (!config.allowSelfLoop && target == i) {
                continue;
            }

            addSuccessor(blocks[i], target, config.maxSucc);
        }
    }

    for (int i = 0; i < config.numBlocks; ++i) {
        sort(blocks[i].succBlocks.begin(), blocks[i].succBlocks.end());
    }
}

// chain CFG를 먼저 만든 뒤 임의의 successor edge를 추가해 random CFG를 생성하는 함수
void generateRandomCFG(vector<Block>& blocks, const Config& config, mt19937& rng) {
    generateChainCFG(blocks, config);

    for (int i = 0; i < config.numBlocks; ++i) {
        int currentSuccCount = (int)blocks[i].succBlocks.size();
        int additionalLimit = config.maxSucc - currentSuccCount;

        if (additionalLimit <= 0) {
            continue;
        }

        uniform_int_distribution<int> addCountDist(0, additionalLimit);
        int addCount = addCountDist(rng);

        uniform_int_distribution<int> succDist(0, config.numBlocks - 1);

        int trial = 0;
        int maxTrial = config.numBlocks * 4;

        while (addCount > 0 && trial < maxTrial) {
            int succ = succDist(rng);

            if (!config.allowSelfLoop && succ == i) {
                trial++;
                continue;
            }

            int beforeSize = (int)blocks[i].succBlocks.size();

            addSuccessor(blocks[i], succ, config.maxSucc);

            int afterSize = (int)blocks[i].succBlocks.size();

            if (afterSize > beforeSize) {
                addCount--;
            }

            trial++;
        }

        sort(blocks[i].succBlocks.begin(), blocks[i].succBlocks.end());
    }
}

// config.cfgType 값에 따라 적절한 CFG 생성 함수를 호출하는 함수
void generateCFG(vector<Block>& blocks, const Config& config, mt19937& rng) {
    if (config.cfgType == "chain") {
        generateChainCFG(blocks, config);
    }
    else if (config.cfgType == "branch") {
        generateBranchCFG(blocks, config, rng);
    }
    else if (config.cfgType == "loop") {
        generateLoopCFG(blocks, config, rng);
    }
    else if (config.cfgType == "random") {
        generateRandomCFG(blocks, config, rng);
    }
    else {
        throw invalid_argument("Unknown cfgType: " + config.cfgType);
    }
}

// 생성된 데이터셋의 변수 인덱스, successor 인덱스, 중복 여부 등을 검사하는 함수
void validateDataset(const vector<Block>& blocks, const Config& config) {
    if ((int)blocks.size() != config.numBlocks) {
        throw runtime_error("Block count mismatch.");
    }

    for (int b = 0; b < config.numBlocks; ++b) {
        set<int> defCheck;
        set<int> useCheck;
        set<int> succCheck;

        for (int v : blocks[b].defVars) {
            if (v < 0 || v >= config.numVars) {
                throw runtime_error("Invalid DEF variable index.");
            }

            if (defCheck.count(v)) {
                throw runtime_error("Duplicate DEF variable.");
            }

            defCheck.insert(v);
        }

        for (int v : blocks[b].useVars) {
            if (v < 0 || v >= config.numVars) {
                throw runtime_error("Invalid USE variable index.");
            }

            if (useCheck.count(v)) {
                throw runtime_error("Duplicate USE variable.");
            }

            if (!config.allowOverlap && defCheck.count(v)) {
                throw runtime_error("USE and DEF overlap is not allowed.");
            }

            useCheck.insert(v);
        }

        for (int s : blocks[b].succBlocks) {
            if (s < 0 || s >= config.numBlocks) {
                throw runtime_error("Invalid successor block index.");
            }

            if (!config.allowSelfLoop && s == b) {
                throw runtime_error("Self-loop is not allowed.");
            }

            if (succCheck.count(s)) {
                throw runtime_error("Duplicate successor.");
            }

            succCheck.insert(s);
        }

        if ((int)blocks[b].succBlocks.size() > config.maxSucc) {
            throw runtime_error("Successor count exceeds maxSucc.");
        }
    }
}

// def 또는 use 줄에 변수 이름 목록을 출력하는 함수
void writeVariableList(
    ofstream& fout,
    const vector<int>& vars,
    const vector<string>& varNames,
    const string& prefix
) {
    fout << prefix;

    for (int v : vars) {
        fout << " " << varNames[v];
    }

    fout << "\n";
}

// succ 줄에 successor 블록 이름 목록을 출력하는 함수
void writeSuccessorList(ofstream& fout, const vector<int>& succBlocks) {
    fout << "succ";

    for (int s : succBlocks) {
        fout << " B" << s;
    }

    fout << "\n";
}

// 최종 Live Variable Analysis 입력 데이터셋 txt 파일을 출력하는 함수
void writeDataset(
    const string& outputFile,
    const vector<string>& varNames,
    const vector<Block>& blocks
) {
    ofstream fout(outputFile);

    if (!fout.is_open()) {
        throw runtime_error("Failed to open output file: " + outputFile + "\nCheck whether the dataset folder exists.");
    }

    fout << "blocks " << blocks.size() << "\n";

    fout << "vars";

    for (const string& name : varNames) {
        fout << " " << name;
    }

    fout << "\n\n";

    for (int i = 0; i < (int)blocks.size(); ++i) {
        fout << "B" << i << ":\n";

        writeVariableList(fout, blocks[i].defVars, varNames, "def");
        writeVariableList(fout, blocks[i].useVars, varNames, "use");
        writeSuccessorList(fout, blocks[i].succBlocks);

        if (i != (int)blocks.size() - 1) {
            fout << "\n";
        }
    }

    fout.close();
}

// 현재 읽어온 설정값과 자동 생성된 output 파일명을 콘솔에 출력하는 함수
void printConfig(const Config& config) {
    cout << "[Config]\n";
    cout << "  blockNum      : " << config.numBlocks << "\n";
    cout << "  variableNum   : " << config.numVars << "\n";
    cout << "  cfgType       : " << config.cfgType << "\n";
    cout << "  defRate       : " << config.defRate << "\n";
    cout << "  useRate       : " << config.useRate << "\n";
    cout << "  loopRate      : " << config.loopRate << "\n";
    cout << "  maxSucc       : " << config.maxSucc << "\n";
    cout << "  seed          : " << config.seed << "\n";
    cout << "  allowOverlap  : " << (config.allowOverlap ? "true" : "false") << "\n";
    cout << "  allowSelfLoop : " << (config.allowSelfLoop ? "true" : "false") << "\n";
    cout << "  outputFile    : " << config.outputFile << "\n";
}

// 프로그램 시작점: config 읽기, 데이터셋 생성, 검증, 파일 출력을 순서대로 수행하는 함수
int main() {
    try {
        string configFile = "config.txt";

        Config config = readConfigFile(configFile);

        validateConfig(config);

        printConfig(config);

        mt19937 rng(config.seed);

        vector<string> varNames = generateVariableNames(config.numVars);

        vector<Block> blocks = generateUseDefSets(config, rng);

        generateCFG(blocks, config, rng);

        validateDataset(blocks, config);

        writeDataset(config.outputFile, varNames, blocks);

        cout << "\n[Success] Dataset generated successfully.\n";
        cout << "Output file: " << config.outputFile << "\n";
    }
    catch (const exception& e) {
        cerr << "\n[Error] " << e.what() << "\n";
        return 1;
    }

    return 0;
}