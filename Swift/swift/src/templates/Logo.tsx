import Image from 'next/image';

import { AppConfig } from '../utils/AppConfig';

type ILogoProps = {
  xl?: boolean;
};

const Logo = (props: ILogoProps) => {
  const fontStyle = props.xl
    ? 'font-semibold text-3xl'
    : 'font-semibold text-xl';

  return (
    <>
      <span className={`text-gray-900 inline-flex items-center ${fontStyle}`}>
        {AppConfig.site_name}
      </span>
      <Image src={'/swift.png'} alt="Home Page" width={100} height={100} />{' '}
    </>
  );
};

export { Logo };
