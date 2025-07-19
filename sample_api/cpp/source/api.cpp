#include <cpprest/containerstream.h>
#include <cpprest/filestream.h>
#include <cpprest/http_listener.h>
#include <cpprest/http_msg.h>
#include <iostream>

using namespace web;
using namespace web::http;
using namespace web::http::experimental::listener;

void handle_get(http_request request)
{
    std::cout << request.to_string() << std::endl;
    request.reply(status_codes::OK, "OK");
}

void handle_post(http_request request)
{
    auto body = request.extract_json();
    body.wait();
    std::cout << body.get().at("id").as_string() << std::endl;
    request.reply(status_codes::OK, body.get());
}
int main(int argc, char* argv[])
{
    http_listener listener(U("http://0.0.0.0:8080/users"));
    listener.support(methods::GET, handle_get);
    listener.support(methods::POST, handle_post);
    listener.open().wait();
    std::cout << "Press ENTER to exit." << std::endl;
    std::string line;
    std::getline(std::cin, line);
    return 0;
}
