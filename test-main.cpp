#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_NOSTDOUT

#include <catch2/catch.hpp>


class out_buff : public std::stringbuf {
	std::FILE *m_stream;
public:
	out_buff(std::FILE *stream) : m_stream(stream) {}

	~out_buff() { pubsync(); }

	int sync() {
		int ret = 0;
		for (unsigned char c : str()) {
			if (putc(c, m_stream) == EOF) {
				ret = -1;
				break;
			}
		}
		// Reset the buffer to avoid printing it multiple times
		str("");
		return ret;
	}
};

namespace Catch {
	std::ostream &cout() {
		static std::ostream ret(new out_buff(stdout));
		return ret;
	}

	std::ostream &clog() {
		static std::ostream ret(new out_buff(stderr));
		return ret;
	}

	std::ostream &cerr() {
		return clog();
	}
}

