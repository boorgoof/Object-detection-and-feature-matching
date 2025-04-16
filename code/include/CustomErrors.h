#ifndef CUSTOM_ERRORS_H
#define CUSTOM_ERRORS_H

#include <stdexcept>
#include <string>

namespace CustomErrors {

    class ImageLoadError : public std::runtime_error {
    public:
        ImageLoadError(const std::string& filename, const std::string& message)
            : std::runtime_error(message), filename_(filename) {}
        const std::string& getFilename() const { return filename_; }
    private:
        std::string filename_;
    };

    class FileNameError : public std::runtime_error {
    public:
        FileNameError(const std::string& filename, const std::string& message)
                : std::runtime_error(message), filename_(filename) {}
        const std::string& getFilename() const { return filename_; }
    private:
        std::string filename_;
    };

    class LabelFormatError : public std::runtime_error {
        public:
            LabelFormatError(const std::string& filename, const std::string& message)
                    : std::runtime_error(message), filename_(filename) {}
            const std::string& getFilename() const { return filename_; }
        private:
            std::string filename_;
        };

    class InputDirectoryError : public std::runtime_error {
    public:
        InputDirectoryError(const std::string& directory, const std::string& message)
            : std::runtime_error(message), directory_(directory) {}
        const std::string& getDirectory() const { return directory_; }
    private:
        std::string directory_;
    };

    class EmptyFolderError : public std::runtime_error {
    public:
        EmptyFolderError(const std::string& directory, const std::string& message)
            : std::runtime_error(message), directory_(directory) {}
        const std::string& getDirectory() const { return directory_; }
    private:
        std::string directory_;
    };

    class OutputDirectoryError : public std::runtime_error {
    public:
        OutputDirectoryError(const std::string& directory, const std::string& message)
            : std::runtime_error(message), directory_(directory) {}
        const std::string& getDirectory() const { return directory_; }
    private:
        std::string directory_;
};

}

#endif
