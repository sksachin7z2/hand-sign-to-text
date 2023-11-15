const ProgressBar = ({ progressPercentage }) => {
    return (
        <>
        {progressPercentage!==0?<div className='h-2 w-full bg-gray-300'>
            <div
                style={{ width: `${progressPercentage}%`}}
                className={`h-full ${
                    progressPercentage < 70 ? 'bg-red-600' : 'bg-green-600'}`}>
            </div>
        </div>: <div></div> }
        </>
    );
};
export default ProgressBar