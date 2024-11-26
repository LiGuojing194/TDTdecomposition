/*
将有向图转为无向图的csr 格式
*/
#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <string>
#include <unordered_map>
using namespace std;

typedef std::pair<int, int> Edge;

int main(int argc, char** argv)
{
    ifstream fin(argv[1]);  ///root/autodl-tmp/TCRTruss32/src/test_data/musae_github.txt
    string outpath = argv[2]; ///root/autodl-tmp/TCRTruss32/src/test_data/musae_github

    int src, dst;
    vector<Edge> edges;

    vector<int> vertices;
    std::unordered_map<int, int> hash_is_exit;
    long cnt = 0;
    long self_loop = 0;
    while(fin >> src >> dst)
    {
        // remove the self loop
        if(src == dst)
        {
            self_loop ++;
            continue;
        }
        if(!hash_is_exit[src]) 
        {
            hash_is_exit[src] = 1;
            vertices.emplace_back(src);
        }
        if(!hash_is_exit[dst]) 
        {
            hash_is_exit[dst] = 1;
            vertices.emplace_back(dst);
        }

        edges.emplace_back(Edge(src, dst));
        edges.emplace_back(Edge(dst, src));
        cnt += 2;
    }  
    cout << "read " << cnt << " edges" << endl;
    cout << "remove " << self_loop << " self loop" << endl;

    fin.close();
    hash_is_exit.clear();
    cout << "read date done" << endl;

    // renumber
    sort(vertices.begin(), vertices.end());
    int vertex_cnt = vertices.size();
    unordered_map<int, int> hash_renumber;
    for(int i = 0; i < vertices.size(); i++)
    {
        hash_renumber[vertices[i]] = i;
    }
    for(auto &e : edges)
    {
        e.first = hash_renumber[e.first];
        e.second = hash_renumber[e.second];
    }
    hash_renumber.clear();
    vertices.clear();
    cout << "renumber done" << endl;

    sort(edges.begin(), edges.end());
    vector<long> vlist(vertex_cnt + 1, 0);
    vector<int> elist;
    int l_src = -1, l_dst = -1;
    cnt = 0;

    for(auto e : edges)
    {
        if(e.first == l_src && e.second == l_dst)
        {
            cnt ++;
            continue;
        }
        vlist[e.first + 1]++;
        elist.emplace_back(e.second);
        l_src = e.first;
        l_dst = e.second;
    }
    for(int i = 1; i < vlist.size(); i++)
    {
        vlist[i] += vlist[i - 1];
    }
    cout << "duplicate edges: " << cnt << endl;
    cout << "vlist size: " << vlist.size() << " elist size: " << elist.size() << endl;
    cout << "generate csr format done" << endl;
    string csr_vlist_path = outpath + "/csr_vlist.bin";
    string csr_elist_path = outpath + "/csr_elist.bin";
    FILE* csr_vlist_file = fopen(csr_vlist_path.c_str(), "w");
    FILE* csr_elist_file = fopen(csr_elist_path.c_str(), "w");
    fwrite(&vlist[0], sizeof(long), vlist.size(), csr_vlist_file);
    fwrite(&elist[0], sizeof(int), elist.size(), csr_elist_file);
    cout << "write csr format done" << std::endl;
}
