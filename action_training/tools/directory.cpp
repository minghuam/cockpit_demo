#include "directory.hpp"
#include "Poco/File.h"
#include "Poco/Path.h"
#include <algorithm>
#include <iostream>

namespace fs{

void ls_all(std::vector<std::string> &files, \
            const std::string& path, \
            bool recursive){
    Poco::File dir(path);
    if(!dir.exists() || !dir.isDirectory()){
        std::cout << "Invalid directory: " << path;
        return;
    }

    std::vector<Poco::File> poco_files;
    dir.list(poco_files);

    std::vector<Poco::File>::iterator it = poco_files.begin();
    while(it != poco_files.end()){
        if(it->isHidden()){
            ++it;
            continue;
        }
        files.push_back(it->path());
        if(it->isDirectory() && recursive){
            ls_all(files, it->path(), recursive);
        }

        it++;
    }

    std::sort(files.begin(), files.end());
}

void ls_files(std::vector<std::string>& files, \
                                  const std::string& path, \
                                  const std::string& extension, \
                                  bool recursive){
    ls_all(files, path, recursive);

    std::vector<std::string>::iterator it = files.begin();
    while(it != files.end()){
        Poco::Path p(*it);
        if(p.getExtension() != extension){
            it = files.erase(it);
        }else{
            ++it;
        }
    }

    std::sort(files.begin(), files.end());
}

std::string join_path(const std::string& parent, const std::string& child){
    Poco::Path poco_parent(parent);
    Poco::Path poco_child(child);
    Poco::Path joined_path = poco_parent.append(poco_child);
    return joined_path.absolute().toString();
}

std::string basename(const std::string& path){
    Poco::Path poco_path(path);
    return poco_path.getFileName();
}

} // end of namespace

