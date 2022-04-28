/* eslint-disable @next/next/link-passhref */
import Link from 'next/link';
import React from 'react';
import mypic from '/public/swift.png'
import Image from "next/image";

const Logo = () => {
    return (
      <Link href="/" >
        <span className="inline-flex items-center justify-center h-20 w-full bg-black hover:bg-black focus:bg-black cursor-pointer">
          <Image
              src={mypic}
              alt="Picture of the author"
              width="350px"
              height="300px"
          />
        </span>
      </Link>
    );
};

export default Logo;
