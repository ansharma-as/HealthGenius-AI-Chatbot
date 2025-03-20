import React, { useState } from "react";

const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false);

  const toggleMenu = () => {
    setIsOpen(!isOpen);
  };

  return (
    <header className="bg-inherit z-20  py-4  fixed top-0 w-full text-black ">
      <div className="container mx-auto flex items-center justify-between">
        {/* Logo Section */}
        <div className="flex items-center space-x-4">
          <a
            href="/"
            className="text-2xl font-bold text-white hover:text-green-500 flex items-center"
          >
            <img src="/Logo11.png" alt="Logo" className="w-10" />
            <span className="ml-2 text-3xl font-bold text-black">
              AI Doctor
            </span>
          </a>
        </div>

        {/* Desktop Navbar */}
        <nav className="hidden lg:flex space-x-6">
          <ul className="flex space-x-6">
            <li>
              <a
                className="text-black hover:text-white transition duration-300 hover:bg-blue-500 py-3 px-4 rounded-full"
                href="/"
              >
                Home
              </a>
            </li>
            <li>
              <a
                className="text-black hover:text-white transition duration-300 hover:bg-blue-500 py-3 px-4 rounded-full"
                href="/contact"
              >
                Contact
              </a>
            </li>
            <li>
              <a
                className="text-black hover:text-white transition duration-300 hover:bg-blue-500 py-3 px-4 rounded-full"
                href="/about"
              >
                About
              </a>
            </li>
            <li>
              <a
                className="bg-blue-500 text-white py-3 px-4 rounded-full hover:text-white hover:bg-blue-800 transition duration-300 "
                href="/chat"
              >
                Get Started
              </a>
            </li>
            {/* <li>
              <a
                className="bg-blue-400 text-white py-3 px-4 rounded-full hover:text-white hover:bg-blue-600 transition duration-300 "
                href="/packet"
              >
                Learn More              </a>
            </li> */}
          </ul>
        </nav>

        {/* Mobile Navbar */}
        <div className="lg:hidden">
          <button
            onClick={toggleMenu}
            className="text-black hover:text-blue-500 focus:outline-none p-4"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              className="h-6 w-6"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M4 6h16M4 12h16M4 18h16"
              />
            </svg>
          </button>
        </div>
      </div>

      {/* Mobile Menu */}
      <div
        className={`${
          isOpen ? "block" : "hidden"
        } lg:hidden bg-slate-300 text-white absolute top-16 left-0 w-full p-8`}
      >
        <ul>
          <li>
            <a
              className="block text-black hover:text-blue-500 transition duration-300"
              href="/"
            >
              Home
            </a>
          </li>
          <li>
            <a
              className="block text-black hover:text-blue-500 transition duration-300 mt-4"
              href="/contact"
            >
              Contact
            </a>
          </li>
          <li>
            <a
              className="block text-black hover:text-blue-500 transition duration-300 mt-4"
              href="/about"
            >
              About Us
            </a>
          </li>
          <li>
            <a
              className="block bg-blue-500 text-white py-2 px-4 rounded mt-6 hover:bg-blue-800 transition duration-300"
              href="/chat"
            >
              Get Started
            </a>
          </li>
        </ul>
      </div>
    </header>
  );
};

export default Navbar;
