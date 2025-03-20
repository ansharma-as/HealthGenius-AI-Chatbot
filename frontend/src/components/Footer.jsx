// import React from 'react'

// function Footer() {
//   return (
//     <div>
//          {/* Footer */}
//       <footer className="bg-transparent text-black py-6 text-center flex items-center justify-evenly">
//         <p className='flex-row'>&copy; 2024 AI Doctor. All rights reserved.</p>
//         <a href="https://github.com/ansharma-as" className='font-bold flex-row'>Designed with 💙 by Ansh</a>
//       </footer>
//     </div>
//   )
// }

// export default Footer


import React, { useState, useEffect } from "react";
import GitHubIcon from "@mui/icons-material/GitHub";
import LinkedInIcon from "@mui/icons-material/LinkedIn";
import InstagramIcon from "@mui/icons-material/Instagram";
import CopyrightIcon from "@mui/icons-material/Copyright";
import MailIcon from '@mui/icons-material/Mail';


function Footer({ darkMode }) {
  const [animateFooter, setAnimateFooter] = useState(false);

  useEffect(() => {
    setTimeout(() => setAnimateFooter(true), 300); // Delay footer animation slightly for a better effect
  }, []);

  return (
    <>
      <div
        className={`bottom-0 left-0 w-full relative ${
          animateFooter ? "translate-y-0" : "translate-y-20"
        } `}
      >
        <div className="w-full  h-auto flex flex-row justify-between items-center p-4 md:p-8 italic bg-inherit relative bottom-0  ">
          <div className="z-0 relative"></div>
          <div className={`transition duration-300 flex items-center hover:animate-pulse ${
                darkMode
                  ? "text-gray-300 hover:text-blue-500"
                  : "text-gray-800 hover:text-blue-600"
              }`}>
            <a
              href="https://www.github.com/ansharma-as"
              target="_blank"
              rel="noopener noreferrer"
            >
              {/* <p className="text-xs sm:text-lg text-black hover:text-blue-400">Developed by Ansh💙</p> */}
            </a>
          </div>

          <div
            className={`transition duration-300 flex items-center hover:animate-pulse text-black ${
              darkMode
                ? "text-gray-300 hover:text-blue-500"
                : "text-gray-800 hover:text-blue-600"
            }`}
          >
            <a
              href="/terms"
              target="_blank"
              rel="noopener noreferrer"
            >
              <span className="mr-1 text-xs sm:text-lg text-black">Copyright</span>
              <CopyrightIcon className=' text-black' fontSize="medium" />
              <span className="ml-1 text-black">2024</span>
            </a>
          </div>

          <div className="flex flex-col lg:flex-row  gap-0 mt-auto md:p-2 gap-x-2 md:gap-x-10">
            <a
              href="mailto:ansharma712.as@gmail.com"
              className={`transition duration-300 flex items-center hover:animate-pulse ${
                darkMode
                  ? "text-gray-300 hover:text-blue-500"
                  : "text-gray-800 hover:text-blue-600"
              }`}
            >
              <MailIcon fontSize="large" />
            </a>
            <a
              href="https://www.instagram.com/ansharma.as/"
              className={`transition duration-300 flex items-center hover:animate-pulse ${
                darkMode
                  ? "text-gray-300 hover:text-blue-500"
                  : "text-gray-800 hover:text-blue-600"
              }`}
            >
              <InstagramIcon fontSize="large" />
            </a>
            <a
              href="https://www.linkedin.com/in/ansharma-as/"
              className={`transition duration-300 flex items-center hover:animate-pulse ${
                darkMode
                  ? "text-gray-300 hover:text-blue-500"
                  : "text-gray-800 hover:text-blue-600"
              }`}
            >
              <LinkedInIcon fontSize="large" />
            </a>
            <a
              href="https://www.github.com/ansharma-as"
              className={`transition duration-300 flex items-center hover:animate-pulse ${
                darkMode
                  ? "text-gray-300 hover:text-blue-500"
                  : "text-gray-800 hover:text-blue-600"
              }`}
            >
              <GitHubIcon fontSize="large" />
            </a>
          </div>
        </div>
      </div>
    </>
  );
}

export default Footer;